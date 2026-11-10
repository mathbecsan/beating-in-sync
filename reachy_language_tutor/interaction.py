"""The heartbeat interaction: a participant's pulse brings Reachy to life.

    waiting    Reachy is asleep (or idle). Nobody is on the sensor.
    waking     A finger landed on the sensor: Reachy wakes up and looks for them.
    syncing    The first seconds: every beat twitches the antennas (and a soft thump plays),
               so the person sees the robot move with their own pulse.
    connected  The pulse is steady: the conversation starts, and the BPM keeps shaping
               how Reachy breathes and sways while it talks.
    searching  The finger came off: Reachy looks around for them. Back on = carry on.
    goodbye    Nobody came back: a short goodbye, then sleep, ready for the next person.

Triggers are finger presence, not BPM (which takes seconds to settle).

Two flows:
    showcase  (default) The camera leads: visitors walk up, Reachy welcomes them and talks
              with them; the heartbeat joins mid-conversation as the climax (finger on =
              syncing, steady = Reachy reacts; finger off = carry on talking). The
              conversation ends on its own when nobody has been around for a while.
    heart     The heartbeat wakes Reachy and starts the conversation (the states above).
"""

import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_language_tutor.main import Tutor

logger = logging.getLogger(__name__)

SYNC_MIN_S = 8.0  # at least this long in the beat-by-beat phase...
SYNC_MAX_S = 16.0  # ...and at most this long, even if the BPM never settles
SEARCH_S = 20.0  # finger gone this long = the person left
GOODBYE_MAX_S = 10.0
BPM_NOTE_DELTA = 8.0  # Reachy remarks on (and asks about) a heart-rate change this big
BPM_NOTE_GAP_S = 20.0
NUDGE_AFTER_S = 10.0  # a pause this long: Reachy asks a new heartbeat question
NUDGE_GAP_S = 18.0
MAX_NUDGES = 3  # in a row without an answer; then it waits (and eventually sleeps)
SAMPLE_RATE = 16000
TOUCH_TALK_S = 5.0  # woken by a touch, no clear pulse yet: start talking anyway after this long
TOUCH_NO_PULSE_S = 20.0  # woken by a touch, no pulse found this long: back to sleep
SHOWCASE_IDLE_S = 60.0  # sleep (heart flow) or doze (camera flow) after this long with nobody talking or on the sensor


def _thump() -> np.ndarray:
    """A soft lub-dub, low and short, so it reads as felt more than heard."""
    t = np.arange(int(0.42 * SAMPLE_RATE)) / SAMPLE_RATE

    def hit(t0: float, f: float, amp: float) -> np.ndarray:
        s = np.clip(t - t0, 0, None)
        return amp * np.sin(2 * np.pi * f * s) * np.exp(-s / 0.045) * (t >= t0)

    x = hit(0.0, 58, 0.55) + hit(0.17, 50, 0.35)
    return x.astype(np.float32)


class Osc:
    """Minimal OSC-over-UDP sender (TouchDesigner, Max, p5 bridges). Fire and forget."""

    def __init__(self, target: str):
        self.addr = None
        if target:
            host, _, port = target.rpartition(":")
            self.addr = (host or "127.0.0.1", int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _pad(b: bytes) -> bytes:
        return b + b"\0" * (4 - len(b) % 4)

    def send(self, address: str, *args) -> None:
        if not self.addr:
            return
        tags, data = ",", b""
        for a in args:
            if isinstance(a, str):
                tags, data = tags + "s", data + self._pad(a.encode())
            elif isinstance(a, (bool, int)):
                tags, data = tags + "i", data + struct.pack(">i", int(a))
            else:
                tags, data = tags + "f", data + struct.pack(">f", float(a))
        try:
            self.sock.sendto(self._pad(address.encode()) + self._pad(tags.encode()) + data, self.addr)
        except OSError:
            pass


class HeartDirector:
    def __init__(self, tutor: "Tutor", osc_target: str = ""):
        self.tutor = tutor
        self.motion = tutor.motion
        self.flow = os.environ.get("REACHY_FLOW", "heart")
        self.drive = True  # in the heart flow: the heart wakes Reachy, starts and ends the conversation
        self.auto_talk = True  # start the conversation once synced
        self.thump = True
        self.state = "waiting"
        self.since = time.monotonic()
        self.osc = Osc(osc_target)
        self._thump = _thump()
        self._told_bpm = 0.0
        self._told_at = 0.0
        self._start_bpm = 0.0
        self._bpm_hist: deque = deque(maxlen=150)  # (t, bpm), ~30 s at 5 Hz
        self._last_nudge = 0.0
        self._nudges = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        """The operator took over (sleep / stop buttons)."""
        with self._lock:
            self._set("waiting")
            self.motion.searching = False
            self.motion.heart_gain = 0.0

    def _set(self, state: str) -> None:
        if state != self.state:
            logger.info("Heart interaction: %s -> %s", self.state, state)
            self.state, self.since = state, time.monotonic()
            self.osc.send("/reachy/state", state)
            label = {
                "waking": "💓 Someone touched the heart sensor. Reachy is waking up",
                "connected": "💞 In sync",
                "searching": "🫳 Finger lifted, Reachy is looking around",
                "goodbye": "👋 Goodbye. Reachy is going to sleep",
            }.get(state)
            if label:
                self.tutor.transcript.append({"role": "event", "text": label, "heart": True})

    # ---- heart events (sensor thread) ----

    def on_heart_event(self, kind: str, info: dict) -> None:
        heart = self.tutor.heart
        if kind == "beat":
            self.osc.send("/heart/beat", float(info.get("bpm") or 0))
            if not heart.finger or self.motion.sleeping:
                return
            self.motion.heartbeat()
            if info.get("bpm"):
                self.motion.heart_bpm = info["bpm"]
            if self.thump and self.state == "syncing" and not self.tutor._speaking():
                try:
                    self.tutor.reachy.media.push_audio_sample(self._thump)
                except Exception:
                    pass
        elif kind == "touch":
            self._touch()
        elif kind == "stable":
            self.osc.send("/heart/bpm", float(info["bpm"]))
        elif kind == "finger_on":
            self.osc.send("/heart/finger", 1)
            self._finger_on()
        elif kind == "finger_off":
            self.osc.send("/heart/finger", 0)
            self._finger_off()

    def _finger_on(self) -> None:
        with self._lock:
            self.motion.searching = False
            if self.flow == "showcase":
                self._showcase_finger_on()
                return
            if not self.drive:
                self.motion.heart_gain = 0.5
                return
            self.tutor.last_activity = time.monotonic()
            if self.state in ("waiting", "goodbye") and self.motion.sleeping:
                self._set("waking")  # a heartbeat starts the visit
                threading.Thread(target=self._wake, daemon=True).start()
            elif self.state in ("waiting", "awake", "goodbye"):
                self._set("syncing")
                self.motion.heart_gain = 1.0
                self.motion.play("curious")
            elif self.state == "connected" and not self._start_bpm:
                # The pulse arrives mid-conversation (they started talking before it was clear):
                # beat-by-beat sync now, and Reachy remarks on it once it's steady (see _converse).
                self.motion.heart_gain = 1.0
                self.motion.play("curious")
            elif self.state == "connected":  # finger back on, mid-conversation
                self.motion.heart_gain = 0.35
                self.motion.play("celebrate")
                self.tutor.note("(They put their finger back on your heart sensor.)", reply=False)

    def _touch(self) -> None:
        """Someone touched the sensor: start waking now; the pulse is confirmed meanwhile."""
        with self._lock:
            if self.flow != "heart" or not self.drive:
                return
            if self.state in ("waiting", "goodbye") and self.motion.sleeping:
                logger.info("Sensor touched: waking up while the pulse is confirmed")
                self.tutor.last_activity = time.monotonic()
                self._set("waking")
                threading.Thread(target=self._wake, daemon=True).start()

    def _showcase_finger_on(self) -> None:
        """Mid-conversation: the pulse arrives. Beats show at once; Reachy reacts once it's steady."""
        if self.state not in ("waiting", "searching", "goodbye"):
            return
        self.tutor.last_activity = time.monotonic()
        self.tutor.rouse("a finger on the heart sensor")
        if self.motion.sleeping:
            self._set("waking")
            threading.Thread(target=self._wake, daemon=True).start()
            return
        self._set("syncing")
        self.motion.heart_gain = 1.0
        self.motion.play("curious")
        self.tutor.note("(Someone just rested a finger on your heart sensor. In a few seconds you'll feel "
                        "their pulse.)", reply=False)

    def _wake(self) -> None:
        if self.motion.sleeping:
            try:
                self.tutor.wake()  # blocking emote, a couple of seconds
            except Exception:
                logger.exception("Wake-up failed; waiting for the next touch")
                with self._lock:
                    self._set("waiting")  # never get stuck in "waking"
                return
        else:
            self.motion.play("curious")
        with self._lock:
            if self.state == "waking":
                if self.tutor.heart.finger or self.flow == "showcase":
                    self._set("syncing")
                    self.motion.heart_gain = 1.0
                else:
                    self._set("awake")  # woken by a touch; syncing starts when the pulse is found

    def _finger_off(self) -> None:
        with self._lock:
            self.motion.heart_gain = 0.0
            self.motion.heart_bpm = 0.0
            if self.flow == "showcase":
                if self.state in ("syncing", "connected", "waking"):
                    self._set("waiting")
                    self.tutor.note("(They took their finger off your heart sensor. Keep talking naturally.)",
                                    reply=False)
                return
            if not self.drive:
                return
            if self.state in ("syncing", "waking"):
                self._set("awake")  # no visit yet; a quiet stretch puts Reachy back to sleep
            elif self.state == "connected":
                # The conversation carries on; it ends with a goodbye or a quiet stretch.
                self.tutor.note("(They lifted their finger off your heart sensor. Keep talking naturally.)",
                                reply=False)

    # ---- timeline (own thread, 5 Hz) ----

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick(time.monotonic())
            except Exception:
                logger.exception("Heart interaction step failed")
            time.sleep(0.2)

    def _tick(self, now: float) -> None:
        heart = self.tutor.heart
        self.osc.send("/heart/live", float(heart.bpm), int(heart.finger))
        with self._lock:
            age = now - self.since
            if self.flow == "showcase":
                self._showcase_tick(now, heart, age)
                return
            if not self.drive:
                return
            if heart.finger:
                self.tutor.last_activity = now
            if self.state == "syncing" and heart.finger:
                if (age >= SYNC_MIN_S and heart.stable) or (age >= SYNC_MAX_S and heart.bpm > 0):
                    self._connect(heart.bpm)
            elif self.state == "connected":
                self._converse(now, heart)
            elif self.state == "goodbye":
                # Let the goodbye finish, then sleep until the next heartbeat.
                if (age > 1.5 and not self.tutor._speaking()) or age > GOODBYE_MAX_S:
                    self._sleep()
                return
            # Woken by a touch, pulse still unclear (weak signal, pressing too hard): don't
            # leave them waiting. Greet them and help them find their pulse.
            if (self.state == "awake" and self.tutor.status == "idle" and age > TOUCH_TALK_S
                    and self.auto_talk and self.tutor.agents.api_key):
                self._set("connected")
                self._start_bpm, self._told_bpm, self._told_at = 0.0, 0.0, now
                self._bpm_hist.clear()
                self._nudges, self._last_nudge = 0, now
                self.tutor.start_session(camera=True, kickoff=(
                    "(Someone touched your heart sensor and woke you up, but you can't feel their pulse "
                    "clearly yet. This is what your camera sees. Greet them warmly and ask their name, and "
                    "gently invite them to rest a fingertip lightly on the glowing light and keep still, so "
                    "you can feel their heartbeat.)"))
                return
            # Woken by a touch but no pulse came (a bump, or pressing too hard): back to sleep.
            if (self.state == "awake" and not heart.finger and self.tutor.status == "idle"
                    and age > TOUCH_NO_PULSE_S):
                logger.info("Touched but no pulse for %.0f s: back to sleep", TOUCH_NO_PULSE_S)
                self._sleep()
                return
            # Quiet for a while (nobody talking, no finger): the visit is over. Sleep.
            if (self.state in ("connected", "awake", "waiting") and not self.motion.sleeping
                    and not self.motion.paused and not self.tutor._speaking()
                    and now - self.tutor.last_activity > SHOWCASE_IDLE_S):
                logger.info("Quiet for %.0f s: going to sleep until the next heartbeat", SHOWCASE_IDLE_S)
                self.tutor.transcript.append({"role": "event", "text": "💤 Quiet for a while. Reachy went to sleep"})
                self._sleep()

    def _set_label(self, text: str) -> None:
        self.tutor.transcript.append({"role": "event", "text": text, "heart": True})

    def _showcase_tick(self, now: float, heart, age: float) -> None:
        t = self.tutor
        if heart.finger:
            t.last_activity = now
        if self.state == "syncing" and heart.finger:
            if (age >= SYNC_MIN_S and heart.stable) or (age >= SYNC_MAX_S and heart.bpm > 0):
                self._set("connected")
                self.motion.heart_gain = 0.35
                self._told_bpm = heart.bpm
                text = (f"(You can feel their heartbeat now: about {heart.bpm:.0f} beats a minute. It's also "
                        f"glowing on the big screen as waves of light. React with delight, briefly, in their language.)")
                if t.status == "connected":
                    t.note(text, reply=True)
                elif t.agents.api_key:
                    t.start_session(kickoff=text)
        elif self.state == "connected" and heart.finger:
            if heart.stable and abs(heart.bpm - self._told_bpm) >= BPM_NOTE_DELTA:
                self._told_bpm = heart.bpm
                t.note(f"(Their heart rate is now about {heart.bpm:.0f} beats per minute.)", reply=False)
        # Nobody has talked for a while: end the conversation and doze until someone comes.
        # (Visitors' speech or a finger counts; Reachy's own talking doesn't, but it never
        # dozes off mid-sentence.)
        if (not self.motion.dozing and not self.motion.sleeping and not self.motion.paused
                and not t._speaking() and now - t.last_activity > SHOWCASE_IDLE_S):
            self._set("waiting")
            t.doze()

    def _heart_story(self, now: float, heart) -> str:
        """Their heartbeat in words, for the agent: now, since the start, and the last ~20 s."""
        if not heart.finger or not heart.bpm:
            return "Their finger is off the sensor right now, so you can't feel their heartbeat"
        past = [b for t, b in self._bpm_hist if now - t > 15]
        trend = ""
        if past:
            d = heart.bpm - past[-1]
            trend = ("; it has been speeding up" if d >= 5 else "; it has been slowing down" if d <= -5
                     else "; it has been steady")
        start = f", it was about {self._start_bpm:.0f} when you met" if self._start_bpm else ""
        return f"Their heart is beating about {heart.bpm:.0f} times a minute{start}{trend}"

    def _converse(self, now: float, heart) -> None:
        """Keep the conversation alive around the heartbeat."""
        t = self.tutor
        if heart.finger and heart.bpm:
            self._bpm_hist.append((now, heart.bpm))
        if t.status != "connected" or t._speaking():
            return
        if t.last_user_speech > self._last_nudge:
            self._nudges = 0  # they answered
        # First steady pulse of a visit that began before the pulse was clear: the big moment.
        if not self._start_bpm and heart.finger and heart.stable:
            self._start_bpm = self._told_bpm = heart.bpm
            self._told_at = self._last_nudge = now
            self.motion.heart_gain = 0.35
            self._set_label("💞 In sync")
            t.note(f"(Now you can feel their heartbeat: about {heart.bpm:.0f} beats a minute, glowing on the big "
                   f"screen as waves of light. React with delight, briefly, and ask them about it.)", reply=True)
            return
        # A real change in their heart rate: say so and ask about it.
        if (heart.finger and heart.stable and abs(heart.bpm - self._told_bpm) >= BPM_NOTE_DELTA
                and now - self._told_at > BPM_NOTE_GAP_S):
            faster = heart.bpm > self._told_bpm
            self._told_bpm, self._told_at = heart.bpm, now
            t.note(f"(Their heart just {'sped up' if faster else 'slowed down'} to about {heart.bpm:.0f} beats "
                   f"a minute. Notice it out loud, playfully, and ask them about it, in their language.)", reply=True)
            self._last_nudge = now
            return
        # A pause: no "are you still there?", but a new question inspired by their heartbeat.
        quiet_since = max(t.last_user_speech, t._speaking_until, self._last_nudge, self.since)
        spaced = self._nudges == 0 or now - self._last_nudge > NUDGE_GAP_S
        if now - quiet_since > NUDGE_AFTER_S and spaced and self._nudges < MAX_NUDGES:
            self._nudges += 1
            self._last_nudge = now
            t.note(f"(A quiet moment. {self._heart_story(now, heart)}. Ask them one new, curious, personal "
                   f"question inspired by their heartbeat or by what they told you, in their language. "
                   f"Don't ask whether they're still there.)", reply=True)

    def _connect(self, bpm: float) -> None:
        self._set("connected")
        self.motion.heart_gain = 0.35  # still visible, but it's the conversation's turn now
        self._told_bpm, self._told_at, self._start_bpm = bpm, time.monotonic(), bpm
        self._bpm_hist.clear()
        self._nudges, self._last_nudge = 0, 0.0
        greet = (f"(Someone's heartbeat just woke you up through your heart sensor: about {bpm:.0f} beats "
                 f"a minute, and it's glowing on the big screen as waves of light. This is what your camera "
                 f"sees. Greet them warmly and expressively, mention feeling their heartbeat, and ask their name.)")
        if self.tutor.status != "idle":
            self.tutor.note(greet, reply=True)
        elif self.auto_talk and self.tutor.agents.api_key:
            try:
                self.tutor.start_session(kickoff=greet, camera=True)
            except Exception as e:
                logger.warning("Could not start the conversation: %s", e)

    def _sleep(self) -> None:
        self._set("waiting")
        self.motion.heart_gain = 0.0
        self.tutor.sleep()

    def request_sleep(self) -> None:
        """The agent's go_to_sleep: the visitor said goodbye. Sleep once the goodbye is spoken."""
        with self._lock:
            if self.flow == "showcase":
                self._set("waiting")
                threading.Thread(target=self._doze_after_goodbye, daemon=True).start()
            else:
                self._set("goodbye")

    def _doze_after_goodbye(self) -> None:
        t0 = time.monotonic()
        while (time.monotonic() - t0 < 1.5 or self.tutor._speaking()) and time.monotonic() - t0 < GOODBYE_MAX_S:
            time.sleep(0.2)
        self.tutor.doze()
