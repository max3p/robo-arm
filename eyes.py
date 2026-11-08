"""Eyes (Phase 3): the USB camera above the claw. Like the ears, they turn what comes in into words for the brain; a
picture reaches it only when it asks for one (`look`), so seeing costs few tokens.

    camera (1280x720, ~30 fps) -> YuNet face detector (OpenCV, CPU, ~10 ms a frame) -> faces: where, how far
                               -> when the arm is still and the picture has changed: Qwen3-VL-2B (GPU, ~4 GB, 1-3 s)
                                  -> a sentence or two, and any hand gesture

The small vision model words a scene well (a thumbs-up, a wave, a person stretching: all right on test photos,
2026-11-08) but its boxes were poor, so where faces are comes from the face detector, and how far from their size: a
face is about FACE_MM wide. Camera calibrated on the arm (2026-11-08): turning the base 10 degrees shifts the picture
143 px, so FOCAL_PX (76 degrees across); not mirrored; it tilts with the forearm, level when shoulder + elbow = 90.

`report(kind, **info)` is called from the eyes' threads: "ready", "seen" (a new description), "arrived" (a face, after
none for AWAY_S), "gesture" (a new one), "error". `pose()` is the arm's feedback (motion.py, T:105) while it is still,
None while it moves: a moving camera's blurred frames are not described, a face it turns toward is not someone
arriving, and where a face is in the room comes from where the camera was. `view()` is what they last saw still, as
words for the brain; `snap()` a picture (JPEG) for the `look` tool, `glance()` a small one of now. `nearest` is the face the body
follows. Frames are never saved.
"""

import glob
import json
import math
import re
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CAMERA = "USB Camera"   # its name in /sys/class/video4linux (not the "PA552" webcam)
WIDTH, HEIGHT = 1280, 720
FOCAL_PX = 825          # at WIDTH: from the base-turn calibration (810 and 841 px)
TILT_DEG = 0            # the camera's tilt above the forearm: not measured, it looks level by eye
VLM = "Qwen/Qwen3-VL-2B-Instruct"  # ~4 GB in fp16, downloaded once to ~/.cache/huggingface
FACES = Path.home() / ".cache/opencv_zoo/face_detection_yunet_2023mar.onnx"  # 230 KB, downloaded once
FACES_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + FACES.name
FACE_SCORE = 0.8        # on test photos, a face in a picture on the wall scored 0.75, a tulip 0.64, a sweater 0.66
FACE_MM = 150           # a face's width, near enough for a distance to +-20%
SETTLE_S = 0.5          # after the arm stops, frames this much later are sharp
EVERY_S = 1             # s between looks at whether the picture changed enough to describe again
CHANGE = 0.02           # ... enough: this share of a 64x36 grey thumbnail changed by over 30 levels (a raised hand)
AWAY_S = 20             # a face after none for this long is someone arriving
LOOK_WIDTH = 800        # px: the picture for `look`, ~600 tokens
GLANCE_WIDTH = 448      # px: the picture sent with what a person says or does, ~190 tokens
PROMPT = ('Reply with JSON only: {"people": <the number of people visible, often 0>, "scene": "<one or two short '
          'sentences on what is in view, and where (left, centre or right; near or far)>", "gesture": "<a hand gesture '
          'a person makes at the camera (thumbs up, wave, pointing, ...), or none>"}')
# Counting people first stopped it making them up: asked "who and what is in view", it saw "a man leaning over a desk"
# at an empty chair (2026-11-08). Saying whose camera it was made it see a "robot pet" in every picture.


def find_camera() -> int:
    for p in sorted(glob.glob("/sys/class/video4linux/video*")):  # each camera has two nodes; index 0 gives pictures
        if Path(p, "name").read_text().startswith(CAMERA) and Path(p, "index").read_text().strip() == "0":
            return int(p.rsplit("video", 1)[1])
    raise RuntimeError(f"no camera named {CAMERA!r}: is it plugged in?")


def open_camera():
    cap = cv2.VideoCapture(find_camera(), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # uncompressed, 720p runs at 5 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open the camera {CAMERA!r}: in use by another program?")
    return cap


def jpeg(frame, width=LOOK_WIDTH) -> bytes:
    small = cv2.resize(frame, (width, round(width * HEIGHT / WIDTH)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def snapshot() -> bytes:
    """One picture, for a program without Eyes running (the MCP server)."""
    cap = open_camera()
    try:
        t = time.monotonic()
        while time.monotonic() - t < 1 or not ok:  # the first frames are dark: the exposure is still settling
            ok, frame = cap.read()
        return jpeg(frame)
    finally:
        cap.release()


def aim(face, fb):
    """A face (x, y, w, h px): its angles right and up from where the camera points (radians), its distance (mm), and
    the point to look_at to face it, from the arm's feedback `fb` (still; None without it). That point holds after
    the arm moves; the rest does not."""
    x, y, w, h = face
    right, up = math.atan((x + w / 2 - WIDTH / 2) / FOCAL_PX), math.atan((HEIGHT / 2 - y - h / 2) / FOCAL_PX)
    d, p = FACE_MM * FOCAL_PX / w, None
    if fb.get("s") is not None:  # the camera is at the claw, turned like the base and tilted like the forearm
        yaw, pitch = fb["b"] - right, math.radians(90 + TILT_DEG) - fb["s"] - fb["e"] + up
        p = (fb["x"] + d * math.cos(pitch) * math.cos(yaw), fb["y"] + d * math.cos(pitch) * math.sin(yaw),
             fb["z"] + d * math.sin(pitch))
    return right, up, d, p


def where(face, fb) -> str:
    """A face in words (see aim)."""
    right, up, d, p = aim(face, fb)
    w = face[2]
    out = (f"a face {abs(math.degrees(right)):.0f} deg {'right' if right > 0 else 'left'}, "
           f"{abs(math.degrees(up)):.0f} deg {'up' if up > 0 else 'down'} from straight ahead, about {d / 1000:.1f} m "
           f"away ({w / WIDTH:.0%} of the picture wide)")
    if p:
        out += f": look_at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) faces it"
    return out


class Eyes:
    def __init__(self, report, pose=lambda: {}):
        self.report, self.pose = report, pose
        self.lock, self.fresh = threading.Lock(), threading.Condition()
        self.frame, self.frame_t, self.faces = None, 0, []  # the latest frame, when it came, its faces (x, y, w, h)
        self.moved = self.face_t = 0                         # when the arm last moved; a face was last seen
        self.said, self.said_t = None, 0                     # the latest description; the frame it was of
        self.found, self.found_t = None, 0                   # the faces in words, the last time the arm was still
        self.nearest = None  # the biggest face seen with the arm still: (when, degrees right and up of the camera's
        # axis, look_at point)
        threading.Thread(target=self._watch, daemon=True).start()

    def view(self) -> str:
        """What the eyes last saw with the arm still, for the brain; "" before they have seen anything."""
        with self.lock:
            said, said_t, found, found_t = self.said, self.said_t, self.found, self.found_t
        old = lambda t: " (before you last moved)" if t < self.moved else ""
        return f"{said}{old(said_t)} Faces{old(found_t)}: {found}." if said and found else ""

    def snap(self) -> bytes:
        """A picture taken SETTLE_S from now: call it once the arm has stopped (the body does). Not "still() for
        SETTLE_S": while the body takes it, still() is False."""
        after = time.monotonic() + SETTLE_S
        with self.fresh:
            ok = self.fresh.wait_for(lambda: self.frame_t > after, 3)
        if not ok:
            raise RuntimeError("the camera is not sending pictures")
        with self.lock:
            return jpeg(self.frame)

    def glance(self) -> bytes | None:
        """The latest frame, small, at once; None before the first."""
        with self.lock:
            return None if self.frame is None else jpeg(self.frame, GLANCE_WIDTH)

    def _watch(self) -> None:
        """Read every frame and find its faces; the describing runs in its own thread, so frames never go stale."""
        try:
            if not FACES.exists():
                FACES.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(FACES_URL, FACES)
            detect = cv2.FaceDetectorYN.create(str(FACES), "", (WIDTH // 2, HEIGHT // 2), FACE_SCORE)
            cap = open_camera()
            self.moved = time.monotonic() + 1.5  # the first frames are dark, the exposure still settling: one was a face
            threading.Thread(target=self._describe, daemon=True).start()
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("the camera stopped sending pictures")
                now, fb = time.monotonic(), self.pose()
                if fb is None:
                    self.moved = now
                _, found = detect.detect(cv2.resize(frame, (WIDTH // 2, HEIGHT // 2)))
                faces = [tuple(2 * float(v) for v in f[:4]) for f in ([] if found is None else found)]
                still = now - self.moved > SETTLE_S
                with self.lock:
                    self.frame, self.frame_t, self.faces = frame, now, faces
                    if still:
                        self.found, self.found_t = "; ".join(where(f, fb) for f in faces) or "none in view", now
                        if faces and fb.get("s") is not None:
                            right, up, _, p = aim(max(faces, key=lambda f: f[2]), fb)
                            self.nearest = now, math.degrees(right), math.degrees(up), p
                with self.fresh:
                    self.fresh.notify_all()
                if faces:
                    if now - self.face_t > AWAY_S and still:
                        self.report("arrived", faces=len(faces))
                    self.face_t = now
        except Exception as e:
            self.report("error", error=f"{type(e).__name__}: {e}")

    def _describe(self) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.float16, device_map="cuda")
            processor = AutoProcessor.from_pretrained(VLM)
            self.report("ready", camera=CAMERA, model=VLM)
            last, gesture = None, None  # the thumbnail and the gesture of the last frame described
            while True:
                time.sleep(EVERY_S)
                with self.lock:
                    frame, frame_t, n = self.frame, self.frame_t, len(self.faces)
                if frame_t - self.moved < SETTLE_S:
                    continue
                thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                if last is not None and n == last[1] and np.mean(cv2.absdiff(thumb, last[0]) > 30) < CHANGE:
                    continue
                last, t = (thumb, n), time.monotonic()
                image = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB))
                messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                         {"type": "text", "text": PROMPT}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                try:
                    said = json.loads(re.search(r"\{.*\}", text, re.S).group())
                    scene, new = str(said["scene"]).strip(), str(said.get("gesture", "")).strip().lower()
                except (AttributeError, ValueError, KeyError, TypeError):  # not the JSON asked for: keep its words
                    scene, new = text.strip(), ""
                new = None if new in ("", "none", "no", "null", "n/a") else new
                with self.lock:
                    self.said, self.said_t = scene, frame_t
                self.report("seen", text=scene, gesture=new, secs=round(time.monotonic() - t, 2))
                if new and new != gesture:
                    self.report("gesture", gesture=new)
                gesture = new
        except Exception as e:
            self.report("error", error=f"{type(e).__name__}: {e}")
