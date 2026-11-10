"""Expressive motion: one 50 Hz loop layers attention, idle life, gestures and speech.

Layers, bottom to top:
  1. Gaze: where Reachy is looking (world frame). Follows the person's face with smooth
     pursuit and small micro-saccades, turns toward voices or movement when nobody is
     in view, and glances around when alone. The body turns on its own when the head
     yaws far (the SDK's automatic body yaw).
  2. Social posture: mirrors the person's head tilt a little, leans back when they get
     very close, leans in when they are far, perks the antennas when someone is there.
  3. Idle breathing.
  4. Gestures the agent triggers (nod, laugh, ...).
  5. Speech wobble from the SDK while Reachy talks.

The participant's heartbeat colors all of it: each beat is a small lub-dub pulse in the
antennas and head, and the BPM sets how fast and how big Reachy breathes and sways
(mirroring the pulse, or breathing slower than it to invite the person to calm down).
"""

import math
import random
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Callable

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

if TYPE_CHECKING:
    from reachy_language_tutor.vision import Vision

# A gesture maps normalized time u in [0, 1] to offsets (roll, pitch, yaw) in degrees
# and antenna offsets (right, left) in radians.
Offsets = tuple[float, float, float, float, float]

CAMERA_LATENCY_S = 0.08  # frame grab -> percept; used to look up where we were looking
YAW_LIMIT = 75.0
PITCH_LIMIT = (-22.0, 18.0)  # up, down


def _env(u: float) -> float:
    """Smooth in/out envelope so gestures start and end at rest."""
    return math.sin(math.pi * u)


def _hold(u: float) -> float:
    """Quick in, hold, quick out."""
    return min(1.0, 4 * u, 4 * (1 - u))


def _nod(u: float) -> Offsets:
    return (0.0, 12.0 * math.sin(4 * math.pi * u) * _env(u), 0.0, 0.0, 0.0)


def _shake(u: float) -> Offsets:
    return (0.0, 0.0, 14.0 * math.sin(4 * math.pi * u) * _env(u), 0.0, 0.0)


def _celebrate(u: float) -> Offsets:
    a = 0.9 * math.sin(8 * math.pi * u) * _env(u)
    return (0.0, -6.0 * _env(u), 0.0, a, -a)


def _think(u: float) -> Offsets:
    e = _hold(u)
    return (14.0 * e, -8.0 * e, 10.0 * e, 0.6 * e, -0.2 * e)


def _curious(u: float) -> Offsets:
    e = _hold(u)
    return (-8.0 * e, 6.0 * e, 0.0, -0.4 * e, -0.4 * e)


def _surprised(u: float) -> Offsets:
    e = min(1.0, 10 * u, 2.5 * (1 - u))  # snap back fast, relax slowly
    return (0.0, -12.0 * e, 0.0, -1.1 * e, 1.1 * e)


def _laugh(u: float) -> Offsets:
    b = math.sin(10 * math.pi * u) * _env(u)
    return (3.0 * _env(u), -8.0 * _env(u) + 4.0 * b, 0.0, 0.4 * b, -0.4 * b)


def _sad(u: float) -> Offsets:
    e = _hold(u)
    return (0.0, 14.0 * e, 0.0, 1.2 * e, -1.2 * e)


def _shy(u: float) -> Offsets:
    e = _hold(u)
    return (-10.0 * e, 10.0 * e, -18.0 * e, 0.5 * e, 0.2 * e)


def _welcome(u: float) -> Offsets:
    """Big hello: perk up, bounce, both antennas waving, a happy head tilt."""
    e = _env(u)
    a = 1.0 * math.sin(7 * math.pi * u) * e
    bounce = 4.0 * abs(math.sin(5 * math.pi * u)) * e
    return (10.0 * math.sin(2 * math.pi * u) * e, -10.0 * e - bounce, 0.0, a - 0.5 * e, -a + 0.5 * e)


def _wave(u: float) -> Offsets:
    a = 1.0 * math.sin(6 * math.pi * u) * _env(u)
    return (6.0 * _env(u), 0.0, 0.0, a, 0.0)


GESTURES: dict[str, tuple[Callable[[float], Offsets], float]] = {
    "nod": (_nod, 1.2),
    "shake_head": (_shake, 1.4),
    "celebrate": (_celebrate, 1.8),
    "think": (_think, 2.5),
    "curious": (_curious, 2.5),
    "surprised": (_surprised, 1.6),
    "laugh": (_laugh, 1.8),
    "sad": (_sad, 2.8),
    "shy": (_shy, 2.4),
    "wave": (_wave, 1.8),
    "welcome": (_welcome, 2.6),
}


def _approach(cur: float, target: float, dt: float, tau: float, max_speed: float) -> float:
    """First-order smoothing toward target with a speed cap (deg/s)."""
    step = (target - cur) * (1 - math.exp(-dt / tau))
    cap = max_speed * dt
    return cur + max(-cap, min(cap, step))


class MotionController:
    def __init__(self, reachy: ReachyMini):
        self.reachy = reachy
        self.mode = "idle"  # idle | listening | speaking
        self.sleeping = False
        self.paused = False
        self.follow = True  # attention on/off
        self.vision: "Vision | None" = None
        self.attention = "idle"  # face | voice | motion | idle (for the UI)
        self._speech = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._gesture: tuple[Callable[[float], Offsets], float, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Gaze state, degrees in the world frame.
        self._gaze = [0.0, 0.0]
        self._target = [0.0, 0.0]
        self._history: deque = deque(maxlen=60)  # (t, yaw, pitch) actually commanded
        self._last_percept = 0.0
        self._last_face = 0.0
        self._attend_until = 0.0
        self._next_glance = 0.0
        self._next_doa = 0.0
        self._saccade = [0.0, 0.0]
        self._next_saccade = 0.0
        self._roll = 0.0
        self._lean = 0.0
        self._perk = 0.0

        # Heartbeat coupling (set by the interaction layer).
        self.heart_bpm = 0.0  # 0 = no heart, use the default calm rhythm
        self.heart_gain = 0.0  # how strongly each beat shows (1 while syncing, subtler later)
        self.heart_mode = "mirror"  # mirror | entrain
        self.searching = False  # finger just left: look around for the person
        self._beat_t = -10.0
        # Dozing (showcase, nobody talking): head a little lowered, antennas drooped, slow deep
        # breaths, but the camera keeps watching and a voice can wake it (see on_voice).
        self.dozing = False
        self.on_voice: Callable[[], None] | None = None
        self._doze = 0.0  # 0 awake .. 1 fully dozing (blended so it drifts off and perks up)
        self._voice_hits = 0
        self._social_until = 0.0  # glancing at another person in a group
        self._next_social = 0.0
        self._breath_phase = 0.0
        self._sway_phase = 0.0
        self._arousal = 0.3

    def start(self) -> None:
        self.reachy.goto_target(head=np.eye(4), antennas=[0.0, 0.0], duration=1.0)
        self._gaze = [0.0, 0.0]
        self._target = [0.0, 0.0]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _halt_loop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._thread = None

    def stop(self) -> None:
        self._halt_loop()
        if not self.sleeping:
            self.reachy.goto_target(head=np.eye(4), antennas=[0.0, 0.0], duration=1.0)

    def pause(self) -> None:
        """Hold still at neutral: no gaze, idle life, gestures or speech wobble."""
        if self.paused or self.sleeping:
            self.paused = True
            return
        self.paused = True
        self._halt_loop()
        self.attention = "idle"
        self.reachy.goto_target(head=np.eye(4), antennas=[0.0, 0.0], duration=1.0)

    def resume(self) -> None:
        if not self.paused:
            return
        self.paused = False
        with self._lock:
            self._gesture = None  # don't replay gestures the agent asked for while frozen
        if not self.sleeping:
            self.start()

    def sleep(self) -> None:
        """Stop all motion and fold into the SDK's sleep pose."""
        if self.sleeping:
            return
        self._halt_loop()
        self.sleeping = True
        for attempt in range(3):
            try:
                self.reachy.goto_sleep()
                return
            except Exception as e:  # e.g. "a move is already running": let it finish
                if attempt == 2:
                    self.sleeping = False
                    if not self.paused:
                        self.start()
                    raise
                time.sleep(1.2)

    def wake(self) -> None:
        """Play the SDK wake-up emote, then resume the idle loop (unless paused)."""
        if not self.sleeping:
            return
        for attempt in range(4):
            try:
                self.reachy.wake_up()
                break
            except Exception:  # e.g. "a move is already running" (still folding into sleep)
                if attempt == 3:
                    raise
                time.sleep(1.0)
        self.sleeping = False
        if not self.paused:
            self.start()

    def set_speech_offsets(self, offsets) -> None:
        """Callback for the SDK head wobbler: (x, y, z, roll, pitch, yaw) in m / rad."""
        self._speech = tuple(offsets)

    def play(self, name: str) -> bool:
        if name not in GESTURES:
            return False
        fn, duration = GESTURES[name]
        with self._lock:
            self._gesture = (fn, time.monotonic(), duration)
        return True

    def heartbeat(self) -> None:
        self._beat_t = time.monotonic()

    def _heart_pulse(self, now: float) -> float:
        """Lub-dub envelope after the last beat, 0..~1."""
        s = now - self._beat_t
        if s < 0 or s > 0.8:
            return 0.0
        lub = math.exp(-s / 0.07)
        dub = 0.55 * math.exp(-(s - 0.17) / 0.08) if s > 0.17 else 0.0
        return self.heart_gain * (lub + dub)

    def look_toward(self, yaw: float, pitch: float = 0.0, hold: float = 2.5) -> None:
        """Briefly attend to a direction relative to the current gaze (degrees)."""
        self._target = [self._gaze[0] + yaw, self._gaze[1] + pitch]
        self._attend_until = time.monotonic() + hold
        self.attention = "voice"

    # ---- attention ----

    def _gaze_at(self, t: float) -> tuple[float, float]:
        """Where we were commanding the gaze at time t (for latency compensation)."""
        best = None
        for sample in self._history:
            if sample[0] <= t:
                best = sample
            else:
                break
        if best is None:
            return self._gaze[0], self._gaze[1]
        return best[1], best[2]

    def _update_attention(self, now: float) -> float:
        """Pick the gaze target; returns the pursuit time constant to use."""
        if self.dozing:
            self.attention = "idle"
            self._target = [0.0, 8.0]  # drooped a little, still facing the room
            self._roll *= 0.97
            self._lean *= 0.97
            self._perk *= 0.98
            self._listen_for_voice(now)
            return 1.5
        p = self.vision.percept if (self.vision and self.follow) else None
        fresh = p is not None and p.stamp > self._last_percept and now - p.stamp < 0.5

        if fresh and p.face:
            self._last_percept = p.stamp
            self._last_face = now
            base_yaw, base_pitch = self._gaze_at(p.stamp - CAMERA_LATENCY_S)
            if now < self._social_until:
                pass  # mid-glance at someone else in the group: hold it
            elif p.others and now > self._next_social:
                # In a group, look from person to person now and then, like a host does.
                oy, op = random.choice(p.others)
                self._target = [base_yaw + oy, base_pitch + op]
                self._social_until = now + random.uniform(1.0, 1.8)
                self._next_social = now + random.uniform(3.0, 6.0)
            # Small dead zone keeps the head calm when the person is roughly centered.
            elif abs(p.err_yaw) > 1.2 or abs(p.err_pitch) > 1.2:
                self._target = [base_yaw + p.err_yaw, base_pitch + p.err_pitch]
            self.attention = "face"
            self._roll = 0.85 * self._roll + 0.15 * max(-12.0, min(12.0, 0.35 * p.roll))
            lean = -1.0 if p.size > 0.35 else (0.6 if p.size < 0.08 else 0.0)
            self._lean = 0.9 * self._lean + 0.1 * lean
            self._perk = 0.9 * self._perk + 0.1
            return 0.16

        if fresh:
            self._last_percept = p.stamp
        self._roll *= 0.97
        self._lean *= 0.97
        since_face = now - self._last_face

        if now < self._attend_until:
            return 0.25
        if since_face < 1.5:
            return 0.3  # hold where they were for a moment; they may come right back

        if self.follow:
            # Turn toward a voice we can hear but not see.
            doa = None
            if self.mode != "speaking" and now > self._next_doa:
                self._next_doa = now + 0.2  # the mic array is polled over USB; 5 Hz is enough
                try:
                    doa = self.reachy.media.get_DoA()
                except Exception:
                    doa = None
            if doa and doa[1]:
                rel = 90.0 - math.degrees(doa[0])  # 0 rad = left, pi/2 = front, pi = right
                if abs(rel) > 15:
                    self.look_toward(0.6 * rel, 0.0, hold=2.0)
                    return 0.25
            # Glance at something that moves.
            if fresh and p.motion is not None:
                base_yaw, base_pitch = self._gaze_at(p.stamp - CAMERA_LATENCY_S)
                self._target = [base_yaw + p.motion[0], base_pitch + p.motion[1]]
                self._attend_until = now + 1.5
                self.attention = "motion"
                return 0.22

        # Alone: glance around now and then, mostly near the front, like idle curiosity.
        self.attention = "idle"
        self._perk *= 0.99
        if now > self._next_glance:
            if self.searching:  # scanning the room for the person who left
                self._target = [random.uniform(-55, 55), random.uniform(-10, 4)]
                self._next_glance = now + random.uniform(1.0, 2.2)
            else:
                self._target = [random.gauss(0, 18), random.uniform(-8, 6)]
                self._next_glance = now + random.uniform(2.5, 6.0)
        return 0.45

    def _listen_for_voice(self, now: float) -> None:
        """While dozing: someone talking near Reachy (mic array voice activity) wakes it."""
        if now < self._next_doa:
            return
        self._next_doa = now + 0.2
        try:
            doa = self.reachy.media.get_DoA()
        except Exception:
            doa = None
        self._voice_hits = self._voice_hits + 1 if (doa and doa[1]) else max(0, self._voice_hits - 1)
        if self._voice_hits >= 4 and self.on_voice:  # ~1 s of speech
            self._voice_hits = 0
            threading.Thread(target=self.on_voice, daemon=True).start()

    def _update_saccade(self, now: float) -> None:
        if self.attention != "face":
            self._saccade = [0.0, 0.0]
            return
        if now > self._next_saccade:
            self._saccade = [random.uniform(-1.4, 1.4), random.uniform(-0.8, 0.8)]
            self._next_saccade = now + random.uniform(0.8, 3.0)

    # ---- loop ----

    def _loop(self) -> None:
        t0 = last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = max(1e-3, now - last)
            last = now
            t = now - t0

            tau = self._update_attention(now)
            self._target[0] = max(-YAW_LIMIT, min(YAW_LIMIT, self._target[0]))
            self._target[1] = max(PITCH_LIMIT[0], min(PITCH_LIMIT[1], self._target[1]))
            speed = 220.0 if self.attention == "face" else 90.0
            self._gaze[0] = _approach(self._gaze[0], self._target[0], dt, tau, speed)
            self._gaze[1] = _approach(self._gaze[1], self._target[1], dt, tau, speed)
            self._update_saccade(now)

            # Breathing and sway follow the heart: ~4 heartbeats per breath. A faster pulse
            # means livelier, bigger motion; a slow one, soft and fluid.
            if self.heart_bpm > 0:
                target_arousal = max(0.0, min(1.0, (self.heart_bpm - 55) / 55))
                breath_hz = self.heart_bpm / 60 / 4 * (0.8 if self.heart_mode == "entrain" else 1.0)
            else:
                target_arousal, breath_hz = 0.3, 0.25
            self._arousal += (target_arousal - self._arousal) * (1 - math.exp(-dt / 3.0))
            ar = self._arousal
            self._doze += ((1.0 if self.dozing else 0.0) - self._doze) * (1 - math.exp(-dt / (2.5 if self.dozing else 0.6)))
            dz = self._doze
            breath_hz = breath_hz * (1 - dz) + 0.12 * dz  # slow, deep breaths when dozing
            self._breath_phase += 2 * math.pi * breath_hz * dt
            self._sway_phase += 2 * math.pi * (0.22 + 0.4 * ar) * dt
            pulse = self._heart_pulse(now)

            yaw = self._gaze[0] + self._saccade[0] + (1 - dz) * (0.8 + 2.5 * ar) * math.sin(0.5 * self._sway_phase)
            pitch = (self._gaze[1] + self._saccade[1]
                     + ((1.2 + 1.6 * ar) * (1 - dz) + 2.8 * dz) * math.sin(self._breath_phase)
                     - 1.5 * pulse)
            roll = self._roll
            # Leaning: back (and chin up) when someone is very close, forward when far.
            x = 0.008 * self._lean if self._lean > 0 else 0.010 * self._lean
            if self._lean < 0:
                pitch += 3.0 * self._lean  # negative pitch = chin up

            listening = 0.35 if self.mode == "listening" else 0.0
            perk = 0.25 * self._perk + listening
            sway = 0.08 + 0.17 * ar
            ant_r = -perk + sway * math.sin(self._sway_phase) - 0.45 * pulse
            ant_l = perk - sway * math.sin(self._sway_phase + 0.6) + 0.45 * pulse
            if dz > 0.01:  # antennas droop, rising and falling a little with each breath
                b = 0.08 * math.sin(self._breath_phase)
                ant_r = ant_r * (1 - dz) + (0.9 + b) * dz
                ant_l = ant_l * (1 - dz) + (-0.9 - b) * dz

            with self._lock:
                g = self._gesture
            if g is not None:
                fn, start, duration = g
                u = (now - start) / duration
                if u >= 1.0:
                    with self._lock:
                        if self._gesture is g:
                            self._gesture = None
                else:
                    dr, dp, dy, da_r, da_l = fn(u)
                    roll, pitch, yaw = roll + dr, pitch + dp, yaw + dy
                    ant_r, ant_l = ant_r + da_r, ant_l + da_l

            self._history.append((now, yaw, pitch))
            sx, sy, sz, s_roll, s_pitch, s_yaw = self._speech if self.mode == "speaking" else (0,) * 6
            pose = create_head_pose(
                x=x + sx, y=sy, z=sz,
                roll=roll + math.degrees(s_roll),
                pitch=pitch + math.degrees(s_pitch),
                yaw=yaw + math.degrees(s_yaw),
                degrees=True, mm=False,
            )
            try:
                self.reachy.set_target(head=pose, antennas=[ant_r, ant_l])
            except Exception:
                pass
            time.sleep(0.02)
