"""The participant's heartbeat: read it from a finger sensor (or fake it) and emit events.

Sources:
  serial  An Arduino / ESP32 on USB. Any of these line formats works, so whatever sketch
          Victoria's board runs is likely fine as is:
            BEAT                  one detected beat
            BPM:72  / BPM=72      a heart-rate estimate
            FINGER:1 / FINGER:0   finger placed / removed
            512                   a raw pulse-signal sample (PulseSensor style)
            IR=51234, BPM=71 ...  MAX30102 example sketch output
          With raw samples, beats are detected here on the laptop. Without FINGER lines,
          a finger counts as present while beats keep coming.
  fake    A steady synthetic heart at a chosen BPM, for developing and as a demo backup.
  off     Nothing.

Nothing is stored: values live in memory only, for the live piece.
"""

import logging
import math
import random
import re
import statistics
import threading
import time
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

BAUD = 115200
REACHY_USB = {(0x1A86, 0x55D3)}  # Reachy Mini's own serial chip, never the sensor
MIN_IBI, MAX_IBI = 60 / 180, 60 / 40  # accept 40-180 bpm
NO_BEAT_FINGER_OFF_S = 3.0
NO_BEAT_TOUCHING_S = 8.0  # same, while the signal shows the finger moving on the sensor  # no beat for this long = finger gone (when the board doesn't say)
MAX30102_NO_FINGER_IR = 50000  # SparkFun example threshold

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class HeartMonitor:
    def __init__(self, on_event: Callable[[str, dict], None], port: str = "", polarity: str = "positive"):
        self.on_event = on_event
        self.polarity = polarity if polarity in ("positive", "negative", "auto") else "positive"
        self.source = "off"
        self.port_override = port
        self.port = ""  # serial port in use
        self.port_error = ""
        self.finger = False
        self.bpm = 0.0
        self.stable = False
        self.beats = 0  # total beats seen (the visuals watch this counter)
        self.last_beat = 0.0
        self.fake_bpm = 72.0
        self.fake_finger = False
        self.signal: deque = deque(maxlen=200)  # recent normalized waveform, for drawing
        self._ibis: deque = deque(maxlen=8)
        self._explicit_finger = False  # board sends FINGER lines
        self._board_bpm = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = None
        self._raw = RawBeatDetector(self.polarity)
        self.raw_lines: deque = deque(maxlen=12)  # last lines from the board, for debugging
        self._line_times: deque = deque(maxlen=500)
        self.raw_tail: deque = deque(maxlen=2500)  # ~10 s of raw readings, for tuning
        self.samples: deque = deque(maxlen=2000)  # (seq, t, value) for the visuals
        self.sample_seq = 0
        self._touch_since = 0.0  # signal active (someone touching the sensor) since
        self._touched = False
        self._candidates: deque = deque(maxlen=5)  # beat times before a finger is trusted
        self._fit: deque = deque(maxlen=6)  # did recent crossings fit the rhythm?
        self._slow_ibis: deque = deque(maxlen=3)  # intervals much longer than the rhythm

    # ---- control ----

    def start(self, source: str = "serial") -> None:
        self.set_source(source)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._close_serial()

    def set_source(self, source: str) -> None:
        if source not in ("serial", "fake", "off"):
            raise ValueError(source)
        if source == self.source:
            return
        self._close_serial()
        self.source = source
        self.port_error = ""
        self._reset(finger=False)

    def set_fake(self, finger: bool | None = None, bpm: float | None = None) -> None:
        if bpm is not None:
            self.fake_bpm = max(40.0, min(180.0, float(bpm)))
        if finger is not None:
            self.fake_finger = finger

    def manual_beat(self) -> None:
        """A tap from the UI (space bar): one beat, and the finger counts as present."""
        self._set_finger(True)
        self._beat(time.monotonic())

    def samples_since(self, seq: int) -> tuple[int, list]:
        """Raw samples newer than `seq`: (latest seq, [[t, value], ...])."""
        with self._lock:
            if not self.samples:
                return self.sample_seq, []
            if seq < self.samples[0][0] - 1 or seq > self.sample_seq:
                seq = max(self.samples[0][0] - 1, self.sample_seq - 1250)  # new client: last 5 s
            out = [[round(t, 4), v] for q, t, v in self.samples if q > seq]
            return self.sample_seq, out

    def snapshot(self) -> dict:
        with self._lock:
            sig = list(self.signal)
        return {
            "source": self.source,
            "port": self.port,
            "port_error": self.port_error,
            "finger": self.finger,
            "bpm": round(self.bpm, 1),
            "stable": self.stable,
            "beats": self.beats,
            "since_beat": round(time.monotonic() - self.last_beat, 3) if self.last_beat else None,
            "fake_bpm": self.fake_bpm,
            "fake_finger": self.fake_finger,
            "signal": [round(v, 3) for v in sig[-120:]],
            "raw_lines": list(self.raw_lines),
            "line_rate": self._line_rate(),
            "sample_rate": round(self._raw._fs, 1),
        }

    def _line_rate(self) -> float:
        lt = list(self._line_times)
        if len(lt) < 2 or time.monotonic() - lt[-1] > 1.0:
            return 0.0
        return round((len(lt) - 1) / (lt[-1] - lt[0]), 1)

    # ---- events ----

    def _reset(self, finger: bool) -> None:
        self._ibis.clear()
        self.bpm, self.stable = 0.0, False
        self._raw = RawBeatDetector(self.polarity)
        self._candidates.clear()
        self._fit.clear()
        self._slow_ibis.clear()
        self._explicit_finger = False
        self._board_bpm = 0.0
        if self.finger and not finger:
            self.finger = False
            self.on_event("finger_off", {})

    def _set_finger(self, on: bool) -> None:
        if on == self.finger:
            return
        self.finger = on
        if not on:
            self._ibis.clear()
            self.bpm, self.stable = 0.0, False
        self.on_event("finger_on" if on else "finger_off", {})

    def _beat(self, now: float, record_ibi: bool = True) -> None:
        if self.last_beat and record_ibi:
            ibi = now - self.last_beat
            if ibi < MIN_IBI * 0.9:
                return  # a double trigger on one beat
            if ibi <= MAX_IBI:
                self._ibis.append(ibi)
        self.last_beat = now
        self.beats += 1
        self._update_bpm()
        self.on_event("beat", {"bpm": self.bpm})

    def _update_bpm(self) -> None:
        if self._board_bpm and not self._ibis:
            self.bpm = self._board_bpm
            return
        if len(self._ibis) < 2:
            return
        recent = list(self._ibis)[-6:]
        med = statistics.median(recent)
        self.bpm = 60.0 / med
        spread = statistics.median(abs(i - med) for i in recent) / med
        was = self.stable
        self.stable = len(self._ibis) >= 5 and spread < 0.12
        if self.stable and not was:
            self.on_event("stable", {"bpm": self.bpm})

    # ---- loop ----

    def _loop(self) -> None:
        next_fake = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if self.source == "fake":
                self._fake_step(now, next_fake)
                if self.fake_finger and now >= next_fake:
                    # A little natural variability so it reads as alive, not a metronome.
                    next_fake = now + 60.0 / self.fake_bpm * random.uniform(0.96, 1.04)
                time.sleep(0.01)
            elif self.source == "serial":
                self._serial_step()
            else:
                time.sleep(0.1)
            # Boards that only send beats: no beats for a while means the finger left.
            # (A finger pressing or shifting still counts as there, for a while longer.)
            if (self.finger and not self._explicit_finger and self.source != "fake" and self.last_beat):
                quiet = now - self.last_beat
                if quiet > NO_BEAT_FINGER_OFF_S and (not self._raw.active or quiet > NO_BEAT_TOUCHING_S):
                    self._set_finger(False)

    def _fake_step(self, now: float, next_fake: float) -> None:
        if self.fake_finger != self.finger:
            self._set_finger(self.fake_finger)
        if self.fake_finger and next_fake and now >= next_fake:
            self._beat(now)
        # Synthetic pulse waveform for the visuals: a sharp upstroke and a slow decay.
        since = now - self.last_beat if self.last_beat else 9
        v = math.exp(-since / 0.12) - 0.35 * math.exp(-max(0, since - 0.25) / 0.2) if self.finger else 0.0
        # Raw-looking samples too (Victoria's --sim shape), so the pulse art works on a fake heart.
        ph = min(1.0, since * self.fake_bpm / 60) if self.finger else 0.0
        shape = math.exp(-((ph - 0.15) / 0.06) ** 2) + 0.35 * math.exp(-((ph - 0.42) / 0.08) ** 2)
        raw = 521 + (130 * shape + random.gauss(0, 3) if self.finger else random.choice((0, 1)))
        with self._lock:
            self.signal.append(v + random.gauss(0, 0.02) if self.finger else 0.0)
            self.sample_seq += 1
            self.samples.append((self.sample_seq, now, int(raw)))

    # ---- serial ----

    def _find_port(self) -> str:
        if self.port_override:
            return self.port_override
        from serial.tools import list_ports

        for p in list_ports.comports():
            if (p.vid, p.pid) in REACHY_USB or p.vid is None:
                continue
            return p.device
        return ""

    def _close_serial(self) -> None:
        s, self._serial = self._serial, None
        if s:
            try:
                s.close()
            except Exception:
                pass
        self.port = ""

    def _serial_step(self) -> None:
        if self._serial is None:
            import serial

            port = self._find_port()
            if not port:
                self.port_error = "No sensor board found on USB"
                time.sleep(1.0)
                return
            try:
                self._serial = serial.Serial(port, BAUD, timeout=0.2)
                self.port, self.port_error = port, ""
                self._raw = RawBeatDetector(self.polarity)  # fresh filters for the fresh stream
                logger.info("Heart sensor on %s", port)
            except Exception as e:
                self.port_error = f"{port}: {e}"
                time.sleep(1.0)
                return
        try:
            line = self._serial.readline().decode("ascii", "ignore").strip()
        except Exception as e:
            logger.warning("Heart sensor disconnected: %s", e)
            self._close_serial()
            self.port_error = "Sensor disconnected"
            self._reset(finger=False)
            return
        if line:
            now = time.monotonic()
            self.raw_lines.append(line)
            self.raw_tail.append(line)
            self._line_times.append(now)
            self._parse(line, now)

    def _parse(self, line: str, now: float) -> None:
        up = line.upper()
        if up.startswith("BEAT"):
            self._set_finger(True)
            self._beat(now)
            return
        if up.startswith("FINGER"):
            nums = _NUM.findall(line)
            self._explicit_finger = True
            self._set_finger(bool(nums) and float(nums[0]) > 0)
            return
        if up.startswith("BPM"):
            nums = _NUM.findall(line)
            if nums and 35 < float(nums[0]) < 200:
                self._board_bpm = float(nums[0])
                if not self._ibis:
                    self.bpm = self._board_bpm
            return
        if "IR=" in up:  # MAX30102 SparkFun-style line
            m = re.search(r"IR=\s*(\d+)", up)
            if m:
                ir = float(m.group(1))
                self._explicit_finger = True
                self._set_finger(ir > MAX30102_NO_FINGER_IR)
                if self.finger:
                    self._raw_sample(ir, now)
            return
        nums = _NUM.findall(line)
        if nums:
            self._raw_sample(float(nums[0]), now)

    def _raw_sample(self, x: float, now: float) -> None:
        """One raw analog reading (any scale, any rate): find beats and the finger."""
        r = self._raw
        beat_t = r.add(x, now)
        # A touch: the signal comes alive (pressing, settling, a first pulse). Reported right
        # away so Reachy can start waking up while the pulse is still being confirmed.
        # (The board resets when the port opens and its first seconds swing wildly: not a touch.)
        if r.touching and not self.finger and r._n > 3 * r._fs:
            self._touch_since = self._touch_since or now
            if not self._touched and now - self._touch_since > 0.5:
                self._touched = True
                self.on_event("touch", {})
        elif not r.touching:
            self._touch_since = 0.0
            if not self.finger:
                self._touched = False
        with self._lock:
            self.sample_seq += 1
            self.samples.append((self.sample_seq, r._clock if r._clock is not None else now, x))
        with self._lock:
            self.signal.append(r.norm)
        if r.noisy:
            # Hum or a floating pin, not a pulse: no beats, and no finger.
            if self.finger and not self._explicit_finger:
                self._set_finger(False)
            return
        if beat_t is None:
            return
        if self._explicit_finger:
            self._beat(beat_t)
            return
        if not self.finger:
            self._try_finger(beat_t)
        else:
            self._rhythm_beat(beat_t)

    def _try_finger(self, t: float) -> None:
        """No finger yet: wait for 5 crossings in a steady, heart-like rhythm."""
        c = self._candidates
        c.append(t)
        if len(c) < c.maxlen:
            return
        ibis = [b - a for a, b in zip(c, list(c)[1:])]
        med = statistics.median(ibis)
        if MIN_IBI <= med <= MAX_IBI and max(abs(i - med) for i in ibis) < 0.15 * med:
            self._ibis.clear()
            self._ibis.extend(ibis)
            self._fit.clear()
            self._set_finger(True)
            self.last_beat = 0.0
            self._beat(t, record_ibi=False)
            self.last_beat = t
            self._update_bpm()

    def _rhythm_beat(self, t: float) -> None:
        """Finger on: accept crossings that fit the rhythm; too many misfits = finger gone."""
        med = statistics.median(self._ibis) if self._ibis else 0.8
        ratio = (t - self.last_beat) / med if self.last_beat else 1.0
        if ratio < 0.65:
            self._fit.append(False)  # an extra crossing between beats (noise, motion)
        elif ratio <= 1.4:
            self._fit.append(True)
            self._slow_ibis.clear()
            self._beat(t)
        else:
            # A pause (missed beats), or the heart really slowed down: if the long
            # intervals keep agreeing, adopt them as the new rhythm.
            ibi = t - self.last_beat
            near_multiple = abs(ratio - round(ratio)) < 0.2
            self._fit.append(near_multiple)
            self._slow_ibis.append(ibi)
            s = list(self._slow_ibis)
            if len(s) == 3 and ibi <= MAX_IBI and max(abs(i - statistics.median(s)) for i in s) < 0.15 * ibi:
                self._ibis.clear()
                self._ibis.extend(s)
                self._slow_ibis.clear()
            self._beat(t, record_ibi=False)
        if len(self._fit) == self._fit.maxlen and sum(not f for f in self._fit) >= 3:
            logger.info("Heart signal lost its rhythm: finger off")
            self._candidates.clear()
            self._set_finger(False)


class RawBeatDetector:
    """Beat detection on a raw pulse waveform, as from analogRead(A0).

    - Sample clock: USB serial delivers lines in bursts, so arrival times are useless for
      filtering. We estimate the board's sample rate and space samples evenly instead.
    - Band-pass: subtract a ~1 s baseline (drift, finger pressure) and low-pass at ~4 Hz
      (mains hum, ADC noise).
    - Polarity: some circuits put the pulse upside down; follow whichever side swings more.
    - Beat: the band-passed signal rises past half its recent amplitude (with re-arm
      hysteresis). Crossings much faster than any heart are counted as noise; many of them
      mean there is no finger, just hum.
    """

    def __init__(self, polarity: str = "positive"):
        # "positive": beats are upward peaks (our A0 circuit). "negative": downward.
        # "auto": decide from the waveform's skew (fine for clearly asymmetric pulses).
        self.polarity_mode = polarity
        self.norm = 0.0
        self.noisy = False
        self._clock: float | None = None
        self._fs = 250.0
        self._arrivals: deque = deque(maxlen=1000)
        self._base = self._lp1 = self._lp2 = 0.0
        self._pos = self._neg = 0.0
        self._polarity = -1.0 if polarity == "negative" else 1.0
        self._m3 = self._a3 = 0.0
        self._above = False
        self._last_cross = -10.0
        self._fast_crossings: deque = deque()
        self._n = 0
        self.artifact = False  # finger pressing / moving: huge swings, rails
        self.touching = False
        self.active = False  # something is touching the sensor (signal not flat)
        self._artifact_until = -1.0
        self._activity = 0.0

    def add(self, x: float, now: float) -> float | None:
        """Feed one sample; returns the beat time if this sample starts a beat."""
        if self._clock is None:
            self._clock = now
            self._base = self._lp1 = self._lp2 = x
            return None
        # Sample rate from the last few seconds of arrivals. The backlog flushed when the
        # port opens arrives all at once, so it ages out instead of skewing the estimate.
        self._arrivals.append(now)
        if self._n % 50 == 0:
            span = self._arrivals[-1] - self._arrivals[0]
            if span > 1.5:
                self._fs = max(20.0, min(1000.0, (len(self._arrivals) - 1) / span))
        dt = 1.0 / max(5.0, self._fs)
        self._clock += dt
        if abs(self._clock - now) > 0.3:  # drifted or the stream stalled: resync
            self._clock = now
        t = self._clock
        self._n += 1

        a_lp = 1 - math.exp(-dt / 0.04)
        self._lp1 += (x - self._lp1) * a_lp
        self._lp2 += (self._lp1 - self._lp2) * a_lp
        self._base += (self._lp2 - self._base) * (1 - math.exp(-dt / 1.0))
        ac = self._lp2 - self._base

        # Motion artifacts: the finger pressing or shifting drives the amp to its rails and
        # swings far beyond a pulse, then recovers slowly. Don't let those inflate the
        # envelope (the real pulse would then look too small for seconds), and take no beats.
        floor = max(2.0, 0.003 * abs(self._base))
        railed = x <= 2 or 1021 <= x <= 1023 or x >= 4093
        env_now = max(self._pos, self._neg)
        if railed or (env_now > 3 * floor and abs(ac) > 3.5 * env_now and self._n > 2 * self._fs):
            self._artifact_until = t + 0.5
        self.artifact = t < self._artifact_until
        self._activity += (abs(ac) - self._activity) * (1 - math.exp(-dt / 1.0))
        self.active = self.artifact or self._activity > 1.5 * floor
        # A touch is a much bigger change than sensor noise (noise here is ~3 counts; a fingertip
        # moves the signal by tens to hundreds), so it has its own, higher bar.
        self.touching = self.artifact or self._activity > max(8.0, 0.015 * abs(self._base))
        decay = math.exp(-dt / 1.5)
        if self.artifact:
            self._pos *= decay
            self._neg *= decay
        else:
            self._pos = max(ac, self._pos * decay)
            self._neg = max(-ac, self._neg * decay)
        # Polarity by shape, not size: a pulse is a sharp peak on one side and a gentle
        # slope on the other, so the signal is skewed toward the peaks (third moment).
        if self.polarity_mode == "auto" and not self.artifact:
            a3 = 1 - math.exp(-dt / 6.0)
            self._m3 += (ac ** 3 - self._m3) * a3
            self._a3 += (abs(ac) ** 3 - self._a3) * a3
            if self._a3 > 0:
                skew = self._m3 / self._a3
                if self._polarity > 0 and skew < -0.3:
                    self._polarity = -1.0
                elif self._polarity < 0 and skew > 0.3:
                    self._polarity = 1.0
        env = self._pos if self._polarity > 0 else self._neg
        v = self._polarity * ac
        self.norm = v / env if env > 1e-9 else 0.0
        if self._n < 0.5 * self._fs:
            return None  # filters still settling

        while self._fast_crossings and t - self._fast_crossings[0] > 2.0:
            self._fast_crossings.popleft()
        self.noisy = len(self._fast_crossings) >= 4

        if self.artifact:
            self._above = False
            return None
        if not self._above and self.norm > 0.5 and env > floor:
            self._above = True
            since = t - self._last_cross
            if since < MIN_IBI * 0.9:  # faster than any heart: noise
                self._fast_crossings.append(t)
                self._last_cross = t
                return None
            self._last_cross = t
            return None if self.noisy else t
        if self._above and self.norm < 0.1:
            self._above = False
        return None
