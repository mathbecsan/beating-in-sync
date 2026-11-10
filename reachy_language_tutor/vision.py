"""Reachy's eyes: face detection, attention targets and social events from the head camera.

Runs in its own thread. Each detection is turned into a gaze target in the world frame
(where the face is, in degrees, relative to the robot), plus a few cues the motion layer
uses to look alive: the person's head tilt, how close they are, and where things move
when nobody is in view. Social events (someone arrives, leaves, waves) go to a callback.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.media.camera_utils import undistort_points
from reachy_mini.vision.face_detector import Face, FaceDetector

logger = logging.getLogger(__name__)

DETECT_WIDTH = 320  # detector input width; small is fast and plenty for faces at 0.3-3 m
PREVIEW_WIDTH = 480
ENGAGE_SIZE = 0.055  # face width / frame width: roughly within 2 m of Reachy
ENGAGE_HOLD_S = 0.8
WAVE_GAP_S = 12.0


@dataclass
class Percept:
    """What Reachy currently sees, in a form the motion layer can act on."""

    face: bool = False
    # Direction of the face relative to the camera axis, degrees (yaw left +, pitch down +).
    err_yaw: float = 0.0
    err_pitch: float = 0.0
    roll: float = 0.0  # the person's head tilt, degrees
    size: float = 0.0  # face width as a fraction of frame width (proximity)
    faces: int = 0
    motion: tuple[float, float] | None = None  # where something moves (err_yaw, err_pitch) when no face
    stamp: float = 0.0  # monotonic time the frame was grabbed
    others: list = field(default_factory=list)  # (err_yaw, err_pitch) of the other faces, for group gaze


@dataclass
class _Presence:
    present: bool = False
    since: float = 0.0
    last_seen: float = 0.0
    last_left: float = 0.0
    close: bool = False
    engaged: bool = False  # someone stepped up close and stayed (fires "engaged" once per visit)
    near_since: float = 0.0
    hand_motion: list = field(default_factory=list)  # (t, side) samples of motion beside the face


class Vision:
    def __init__(self, reachy: ReachyMini, on_event: Callable[[str, dict], None]):
        self.reachy = reachy
        self.on_event = on_event
        self.percept = Percept()
        self.enabled = True
        self._detector: FaceDetector | None = None
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None  # latest annotated preview
        self._frame: np.ndarray | None = None  # latest raw frame (for the agent's "look")
        self._prev_gray: np.ndarray | None = None
        self._presence = _Presence()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self._last_wave = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self.percept = Percept()

    # ---- outputs ----

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def snapshot_jpeg(self, width: int = 1024) -> bytes | None:
        """A clean, reasonably sized frame for the LLM to look at."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    # ---- loop ----

    def _loop(self) -> None:
        try:
            self._detector = FaceDetector(score_threshold=0.6)
        except Exception:
            logger.exception("Could not load the face detector; vision disabled")
            return
        last = time.monotonic()
        while not self._stop.is_set():
            if not self.enabled:
                # Drop the last frames too, so nothing stale is shown or sent to the agent.
                self.percept = Percept()
                self.fps = 0.0
                self._prev_gray = None
                with self._lock:
                    self._frame = self._jpeg = None
                time.sleep(0.2)
                continue
            frame = self.reachy.media.get_frame()
            stamp = time.monotonic()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                self._process(frame, stamp)
            except Exception:
                logger.exception("Vision step failed")
                time.sleep(0.2)
            dt = stamp - last
            last = stamp
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 / dt
            time.sleep(0.03)  # ~15-20 Hz is plenty for attention and keeps CPU low

    def _angles(self, u: float, v: float, scale: float) -> tuple[float, float]:
        """Pixel in the detector image -> (yaw, pitch) error of that point, degrees."""
        cam = self.reachy.media.camera
        x_n, y_n = undistort_points(u / scale, v / scale, cam.K, cam.D)
        # Camera optical frame: x right, y down, z forward. Head frame: x forward, y left, z up.
        return math.degrees(math.atan2(-x_n, 1.0)), math.degrees(math.atan2(y_n, 1.0))

    def _process(self, frame: np.ndarray, stamp: float) -> None:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        faces = self._detector.detect(small)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        motion_mask = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        self._prev_gray = gray

        p = Percept(stamp=stamp, faces=len(faces))
        target = self._pick(faces)
        for f in faces:
            if f is not target:
                x0, y0, bw, bh = f.bbox
                p.others.append(self._angles(x0 + bw / 2, y0 + bh * 0.45, scale))
        if target is not None:
            x0, y0, bw, bh = target.bbox
            cu, cv_ = x0 + bw / 2, y0 + bh * 0.45  # a bit above center: look at the eyes
            p.face = True
            p.err_yaw, p.err_pitch = self._angles(cu, cv_, scale)
            (rx, ry), (lx, ly) = target.right_eye, target.left_eye
            p.roll = math.degrees(math.atan2(ly - ry, lx - rx)) if lx != rx else 0.0
            if abs(p.roll) > 90:
                p.roll -= math.copysign(180, p.roll)
            p.size = bw / small.shape[1]
            if motion_mask is not None:
                self._track_hand_motion(motion_mask, target, stamp)
        elif motion_mask is not None:
            p.motion = self._motion_target(motion_mask, scale)

        self.percept = p
        self._update_presence(p, stamp)
        preview = self._annotate(frame, faces, target, scale)
        with self._lock:
            self._frame = frame
            self._jpeg = preview

    def _pick(self, faces: list[Face]) -> Face | None:
        """Prefer the face we were already looking at; otherwise the biggest (closest) one."""
        if not faces:
            return None
        prev = self.percept
        if prev.face and len(faces) > 1:
            def score(f: Face) -> float:
                x0, y0, bw, bh = f.bbox
                return bw * bh - 4000 * abs(bw / DETECT_WIDTH - prev.size)
            return max(faces, key=score)
        return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

    def _motion_target(self, mask: np.ndarray, scale: float) -> tuple[float, float] | None:
        moving = cv2.countNonZero(mask)
        # Too little is noise; too much means our own head is moving and the whole image shifts.
        if moving < mask.size * 0.004 or moving > mask.size * 0.2:
            return None
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] == 0:
            return None
        return self._angles(m["m10"] / m["m00"], m["m01"] / m["m00"], scale)

    def _track_hand_motion(self, mask: np.ndarray, face: Face, stamp: float) -> None:
        """A wave is a raised hand moving beside the head, over and over.

        Talking people move their shoulders and hands, and Reachy's own head motion shifts
        the whole image, so: only motion at head height and a little away from the face,
        only when the rest of the image is calm, sustained, and at most one every WAVE_GAP_S.
        """
        if cv2.countNonZero(mask) > mask.size * 0.06:
            return  # ego-motion: the head is turning, not a hand waving
        if stamp - self._last_wave < WAVE_GAP_S:
            return
        x0, y0, bw, bh = face.bbox
        H, W = mask.shape
        top = int(max(0, y0 - bh * 1.2))
        bot = int(min(H, y0 + bh * 0.6))  # head height, not chest-level talking hands
        left = mask[top:bot, int(max(0, x0 - bw * 2.4)):int(max(0, x0 - bw * 0.6))]
        right = mask[top:bot, int(min(W, x0 + bw * 1.6)):int(min(W, x0 + bw * 3.4))]
        area = max(1, bw * bh)
        side = None
        for name, region in (("left", left), ("right", right)):
            if region.size and cv2.countNonZero(region) > area * 0.25:
                side = name
        pres = self._presence
        pres.hand_motion = [(t, s) for t, s in pres.hand_motion if stamp - t < 1.6]
        if side:
            pres.hand_motion.append((stamp, side))
        if len(pres.hand_motion) >= 9:
            pres.hand_motion.clear()
            self._last_wave = stamp
            self.on_event("wave", {})

    def _update_presence(self, p: Percept, stamp: float) -> None:
        pres = self._presence
        if p.face:
            pres.last_seen = stamp
            if not pres.present:
                pres.present = True
                pres.since = stamp
                away = stamp - pres.last_left if pres.last_left else float("inf")
                self.on_event("person_arrived", {"away_s": away, "faces": p.faces})
            # Stepping up: a face big enough to be in front of Reachy (not across the room),
            # held for a moment, so passers-by don't trigger a welcome.
            if p.size >= ENGAGE_SIZE:
                pres.near_since = pres.near_since or stamp
                if not pres.engaged and stamp - pres.near_since >= ENGAGE_HOLD_S:
                    pres.engaged = True
                    self.on_event("engaged", {"faces": p.faces, "size": p.size})
            else:
                pres.near_since = 0.0
            if p.size > 0.42 and not pres.close and stamp - pres.since > 1.0:
                pres.close = True
                self.on_event("very_close", {})
            elif p.size < 0.3:
                pres.close = False
        elif pres.present and stamp - pres.last_seen > 6.0:  # brief misses (turning, occlusion) are not leaving
            pres.present = False
            pres.engaged = False
            pres.near_since = 0.0
            pres.last_left = stamp
            self.on_event("person_left", {"stayed_s": stamp - pres.since})

    def _annotate(self, frame: np.ndarray, faces: list[Face], target: Face | None, scale: float) -> bytes | None:
        h, w = frame.shape[:2]
        view = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        k = PREVIEW_WIDTH / w / scale  # detector pixels -> preview pixels
        for f in faces:
            x0, y0, bw, bh = (int(v * k) for v in f.bbox)
            color = (80, 220, 120) if f is target else (200, 200, 200)
            cv2.rectangle(view, (x0, y0), (x0 + bw, y0 + bh), color, 2)
        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None
