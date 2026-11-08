"""Ears (Phase 2.6): always listening on the Blue Snowball. A voice detector cuts the sound into utterances, Whisper
turns each one into text locally on the GPU, and arm.py hands the text to the brain like a typed message.

    mic (48 kHz, via PipeWire) -> 16 kHz -> Silero voice detector, 32 ms frames -> SILENCE_S of quiet ends it
        -> faster-whisper (STT_MODEL, CUDA) -> report("heard", text=...)

Only what the detector calls speech is ever transcribed: given 3 s of room silence, Whisper wrote "Thank you." (with
a no-speech probability of 0.0), so it must never see silence or hum. Every utterance, kept or dropped, is saved as a
.wav in `clips`, so a test run can be checked afterwards.
`report(kind, **info)` is called from the ears' own thread with kind "ready", "speech" (someone started talking),
"heard", "dropped" or "error".
"""

import ctypes
import glob
import queue
import threading
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd

MIC = "Blue Snowball Mono"   # PipeWire's name for it; `python -m sounddevice` lists them
MIC_RATE, RATE, FRAME = 48000, 16000, 512  # the mic opens at 48 kHz only; the detector and Whisper take 16 kHz
STT_MODEL, LANGUAGE = "large-v3-turbo", "en"  # ~1.6 GB, downloaded once to ~/.cache/huggingface; "small.en" is faster
START_P, END_P = 0.5, 0.35  # the detector's speech probability: speech starts above START_P, ends below END_P
SILENCE_S = 0.5     # this much quiet ends an utterance
PREROLL_S = 0.3     # kept from before the start, so the first syllable isn't clipped
MIN_SPEECH_S = 0.3  # less speech than this (a clunk, a cough) is dropped without transcribing
MAX_SPEECH_S = 15   # longer speech is cut here, so the brain isn't kept waiting
MIN_LOGPROB = -1.0  # Whisper's mean log-probability per word piece: below this it is likely making words up


def load_cuda_libs() -> None:
    """CTranslate2 needs cuBLAS and cuDNN. They come from pip (nvidia-*-cu12), which is not on the library path, so
    load them up front; the ones that need another first are retried."""
    import nvidia
    pending = sorted(p for d in nvidia.__path__ for p in glob.glob(f"{d}/cu*/lib/libcu*.so.*"))
    for _ in range(len(pending)):
        for p in list(pending):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                pending.remove(p)
            except OSError:
                pass


class Ears:
    def __init__(self, report, clips):
        self.report, self.clips, self.n = report, clips, 0
        self.audio = queue.Queue()  # 32 ms frames at 16 kHz; None ends listening
        threading.Thread(target=self._run, daemon=True).start()

    def close(self) -> None:
        self.audio.put(None)

    def _run(self) -> None:
        try:
            load_cuda_libs()
            from faster_whisper import WhisperModel
            from faster_whisper.vad import get_vad_model
            self.whisper = WhisperModel(STT_MODEL, device="cuda", compute_type="float16")
            self.vad = get_vad_model().session
            list(self.whisper.transcribe(np.zeros(RATE, "float32"))[0])  # warm up: the first run is ~0.3 s slower
            put = lambda data, *_: self.audio.put(data[:, 0].reshape(-1, 3).mean(1))  # 48 -> 16 kHz
            with sd.InputStream(MIC_RATE, FRAME * 3, MIC, 1, "float32", callback=put):
                self.report("ready", mic=MIC, model=STT_MODEL)
                self.listen(self._frames())
        except Exception as e:
            self.report("error", error=f"{type(e).__name__}: {e}")

    def _frames(self):
        while (frame := self.audio.get()) is not None:  # not iter(get, None): that compares numpy arrays with ==
            yield frame

    def listen(self, frames) -> None:
        """Cut a stream of FRAME-sample 16 kHz frames into utterances, and transcribe each one."""
        h = c = np.zeros((1, 1, 128), "float32")  # the detector's memory, carried from frame to frame
        context = np.zeros(64, "float32")
        before = deque(maxlen=round(PREROLL_S * RATE / FRAME))
        speech = None  # the frames of the utterance in progress
        for frame in frames:
            p, h, c = self.vad.run(None, {"input": np.concatenate([context, frame])[None], "h": h, "c": c})
            p, context = p[0], frame[-64:]
            if speech is None:
                before.append(frame)
                if p > START_P:
                    speech, voiced, quiet = list(before), 0, 0
                    self.report("speech")
                continue
            speech.append(frame)
            voiced, quiet = (voiced + 1, 0) if p >= END_P else (voiced, quiet + 1)
            if quiet * FRAME >= SILENCE_S * RATE or len(speech) * FRAME >= MAX_SPEECH_S * RATE:
                self.utterance(np.concatenate(speech), voiced * FRAME / RATE)
                speech = None
                before.clear()

    def utterance(self, audio, voiced_s) -> None:
        self.n += 1
        info = {"clip": f"{self.n:03d}.wav", "speech_s": round(voiced_s, 2), "clip_s": round(len(audio) / RATE, 2)}
        self.clips.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.clips / info["clip"]), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        if voiced_s < MIN_SPEECH_S:
            return self.report("dropped", why="too short", **info)
        t = time.monotonic()
        segments, _ = self.whisper.transcribe(audio, language=LANGUAGE, beam_size=1, without_timestamps=True,
                                              condition_on_previous_text=False)
        segments = list(segments)
        text = "".join(s.text for s in segments).strip()
        info |= {"text": text, "stt_s": round(time.monotonic() - t, 2),
                 "logprob": round(min((s.avg_logprob for s in segments), default=0), 2)}
        if not text:
            self.report("dropped", why="no words", **info)
        elif info["logprob"] < MIN_LOGPROB:
            self.report("dropped", why="unsure, likely made up", **info)
        else:
            self.report("heard", **info)
