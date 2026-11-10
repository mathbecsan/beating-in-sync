#!/usr/bin/env python3
# Pulse.py
# Author: Victoria Virgen
# This file takes the serial stream from the Arduino analog read, cleans it up, then
# creates and displays the pulse waveform, in real time, as an artistic visualization.
import argparse
import colorsys
import collections
import math
import random
import threading
import time
import numpy as np
import pygame

# ----------------------------------------------------------------- settings
WINDOW_S = 5.0          # seconds of signal history kept
VIEW_S = 4.0            # seconds of history spread across the screen width
N_LINES = 20            # number of ribbons
N_POINTS = 110          # points per ribbon
ADC_LO, ADC_HI = 3, 1020  # readings outside this are clamped spikes -> dropped
BEAT_THRESHOLD = 0.65   # normalized level that counts as a beat on the upswing
BEAT_REFRACTORY_S = 0.35

# look
SUPPORT_BRIGHTNESS = 0.35   # 0..1, how bright the background ribbons are
SINE_AMOUNT = 0.02          # decorative wobble size (was 0.07); 0 = pulse only
HERO_SMOOTH = 7             # smoothing of the main line (small = keeps pulse detail)
HERO_WIDTH = 3              # thickness of the main line in pixels


# --------------------------------------------------------- sample collection
class SampleBuffer:
    """Thread-safe rolling buffer of (time, value)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t = collections.deque()
        self.v = collections.deque()
        self.rejected = 0
        self.reject_times = collections.deque(maxlen=400)

    def push(self, value, t=None):
        t = time.monotonic() if t is None else t
        if value <= ADC_LO or value >= ADC_HI:
            with self.lock:
                self.rejected += 1  # clamped motion spike: drop it
                self.reject_times.append(t)
            return
        with self.lock:
            self.t.append(t)
            self.v.append(value)
            cutoff = t - WINDOW_S
            while self.t and self.t[0] < cutoff:
                self.t.popleft()
                self.v.popleft()

    def reject_snapshot(self):
        with self.lock:
            return list(self.reject_times)

    def snapshot(self):
        with self.lock:
            return np.array(self.t), np.array(self.v, dtype=float)


def serial_worker(port, baud, buf, stop):
    import serial
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # Nano resets when the port opens
    ser.reset_input_buffer()
    while not stop.is_set():
        line = ser.readline()
        try:
            buf.push(int(line.strip()))
        except ValueError:
            pass  # partial/garbled line, ignore
    ser.close()


def sim_worker(buf, stop, bpm=74):
    """Fake PPG: pulse shape + noise + occasional clamped motion spike."""
    rate = 250
    phase = 0.0
    last = time.monotonic()
    next_spike = time.monotonic() + 12
    spike_until = 0
    while not stop.is_set():
        now = time.monotonic()
        phase += (bpm / 60.0) * (now - last) * (1 + 0.05 * math.sin(now * 0.3))
        last = now
        phase %= 1.0
        # systolic peak + dicrotic bump
        s = math.exp(-((phase - 0.15) / 0.06) ** 2) + 0.35 * math.exp(-((phase - 0.42) / 0.08) ** 2)
        value = 500 + 130 * s + random.gauss(0, 3) + 25 * math.sin(now * 0.4)
        if now > next_spike:
            spike_until = now + 0.3
            next_spike = now + random.uniform(10, 20)
        if now < spike_until:
            value = random.choice([0, 1023])
        buf.push(int(value))
        time.sleep(1.0 / rate)


# ----------------------------------------------------------------- analysis
class Analyzer:
    """Auto-ranges the signal to 0..1 and detects beats."""

    def __init__(self):
        self.lo = None
        self.hi = None
        self.last_beat = -1e9
        self.intervals = collections.deque(maxlen=6)
        self.bpm = 0.0
        self.beat_times = collections.deque(maxlen=30)
        self.energy = 0.0   # jumps to 1 on each beat, decays after
        self.has_signal = False

    def update(self, ts, vs, now, dt):
        self.energy *= math.exp(-dt / 0.35)
        self.has_signal = False
        if len(vs) < 100 or ts[-1] - ts[0] < 2.0:
            return None

        padded = np.pad(vs, (7, 7), mode="edge")  # pad so the newest samples are not dragged toward 0
        sm = np.convolve(padded, np.ones(15) / 15, mode="valid")  # ~60 ms smoothing at 250 Hz
        lo, hi = np.percentile(sm, [2, 99.5])
        if hi - lo < 4:          # basically flat: finger is off the sensor
            return None
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:                     # slow tracking so the range doesn't jitter
            self.lo += (lo - self.lo) * 0.05
            self.hi += (hi - self.hi) * 0.05
        norm = np.clip((sm - self.lo) / (self.hi - self.lo), 0, 1.1)
        self.has_signal = True

        # beat detection: upward crossings of the threshold
        idx = np.where((norm[:-1] < BEAT_THRESHOLD) & (norm[1:] >= BEAT_THRESHOLD))[0]
        for i in idx:
            tb = ts[i + 1]
            if tb - self.last_beat > BEAT_REFRACTORY_S:
                if self.last_beat > 0:
                    self.intervals.append(tb - self.last_beat)
                self.last_beat = tb
                self.beat_times.append(tb)
                if now - tb < 0.25:
                    self.energy = 1.0
        if len(self.intervals) >= 3:
            med = float(np.median(self.intervals))
            if 0.3 < med < 2.0:
                self.bpm = 60.0 / med
        return ts, norm


# ---------------------------------------------------------------- rendering
def make_palette(n):
    """pink -> purple -> blue -> cyan -> green, left to right."""
    cols = []
    for j in range(n):
        x = j / (n - 1)
        h = 0.92 - 0.54 * x
        cols.append(colorsys.hsv_to_rgb(h, 0.75, 1.0))
    return np.array(cols) * 255


def render(canvas, t_now, analysis, energy, palette):
    w, h = canvas.get_size()
    canvas.fill((0, 0, 0))
    cy = h * 0.5
    xs = np.linspace(0, 1, N_POINTS)
    px = xs * w
    env = np.sin(np.pi * xs) ** 0.4          # taper toward the edges

    def signal_at(delay_s, k):
        """Your pulse (-0.5..0.5) across the screen, delayed and smoothed."""
        if analysis is None:
            return np.zeros(N_POINTS)
        ts, norm = analysis
        times = t_now - VIEW_S + xs * VIEW_S - delay_s
        sig = np.interp(times, ts, norm, left=0.5, right=norm[-1]) - 0.5
        sig = np.pad(sig, (k, k), mode="edge")
        return np.convolve(sig, np.ones(k) / k, mode="same")[k:-k]

    # --- supporting ribbons: dim, delayed, smoothed echoes + decorative sines
    for i in range(N_LINES):
        u = i / (N_LINES - 1)
        mid = 1 - abs(2 * u - 1)             # 0 at outer lines, 1 in the middle
        sig = signal_at(u * 1.2, 7 + int(u * 22))

        freq = 0.7 + 1.3 * ((i * 0.618) % 1.0)
        phase = i * 0.9
        speed = 0.25 + 0.35 * ((i * 0.37) % 1.0)
        sine = np.sin(2 * np.pi * freq * xs + phase + t_now * speed)
        sine *= h * SINE_AMOUNT * (0.6 + 0.9 * energy)

        y = cy - env * (sig * h * 0.55 * (0.35 + 0.65 * mid) + sine + (u - 0.5) * h * 0.14)

        b = min(1.0, SUPPORT_BRIGHTNESS * (0.3 + 0.7 * mid) * (0.8 + 0.4 * energy))
        cols = (palette * b).astype(int).tolist()
        for j in range(N_POINTS - 1):
            pygame.draw.line(canvas, cols[j], (px[j], y[j]), (px[j + 1], y[j + 1]), 1)

    # --- hero line: live pulse, barely smoothed, no delay, thick and bright
    sig = signal_at(0.0, HERO_SMOOTH)
    wobble = np.sin(2 * np.pi * 1.1 * xs + t_now * 0.4) * h * 0.015
    hero_env = 0.35 + 0.65 * env             # keeps the newest beat visible at the right edge
    y = cy - hero_env * (sig * h * 0.55 + wobble)
    cols = np.minimum(255, palette * (0.9 + 0.1 * energy) + 110 * energy).astype(int).tolist()
    for j in range(N_POINTS - 1):
        pygame.draw.line(canvas, cols[j], (px[j], y[j]), (px[j + 1], y[j + 1]), HERO_WIDTH)


def draw_signal_panel(screen, font, buf, ts, vs, analysis, analyzer, now):
    """Bottom strip showing what the code actually sees."""
    w, h = screen.get_size()
    ph = int(h * 0.27)
    half = ph // 2
    panel = pygame.Surface((w, ph))
    panel.fill((12, 12, 18))

    def tx(t):
        return (t - (now - WINDOW_S)) / WINDOW_S * (w - 1)

    pygame.draw.line(panel, (40, 40, 55), (0, half), (w, half), 1)

    # row 1: raw ADC values (spikes already removed)
    if len(vs) > 2:
        lo, hi = vs.min(), vs.max()
        span = max(hi - lo, 1.0)
        pts = [(tx(t), half - 8 - (v - lo) / span * (half - 24)) for t, v in zip(ts, vs)]
        pygame.draw.lines(panel, (150, 150, 165), False, pts, 1)
        label = f"raw ADC after spike removal   min {lo:.0f}  max {hi:.0f}  range {hi - lo:.0f}"
    else:
        label = "raw ADC: no data yet"
    panel.blit(font.render(label, True, (120, 120, 140)), (8, 3))
    for t in buf.reject_snapshot():            # red ticks where spikes were dropped
        x = tx(t)
        if 0 <= x < w:
            pygame.draw.line(panel, (230, 60, 60), (x, 2), (x, 8), 1)

    # row 2: normalized signal, threshold, beats
    top2, bot2 = half + 18, ph - 6
    thr_y = bot2 - BEAT_THRESHOLD * (bot2 - top2)
    pygame.draw.line(panel, (70, 70, 40), (0, thr_y), (w, thr_y), 1)
    if analysis is not None:
        nts, norm = analysis
        pts = [(tx(t), bot2 - min(v, 1.0) * (bot2 - top2)) for t, v in zip(nts, norm)]
        pygame.draw.lines(panel, (90, 220, 230), False, pts, 2)
    for tb in analyzer.beat_times:
        x = tx(tb)
        if 0 <= x < w:
            pygame.draw.line(panel, (255, 90, 190), (x, half + 14), (x, ph), 2)
    panel.blit(font.render("normalized 0-1 (cyan)   beat threshold (yellow line)   detected beats (pink)",
                           True, (120, 120, 140)), (8, half + 3))
    screen.blit(panel, (0, h - ph))


def apply_glow(screen, canvas):
    w, h = canvas.get_size()
    screen.blit(canvas, (0, 0))
    for div, repeat in ((4, 1), (12, 2)):
        small = pygame.transform.smoothscale(canvas, (max(1, w // div), max(1, h // div)))
        glow = pygame.transform.smoothscale(small, (w, h))
        for _ in range(repeat):
            screen.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port of the Arduino")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sim", action="store_true", help="use a simulated pulse")
    ap.add_argument("--max-seconds", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--snapshot", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.sim and not args.port:
        ap.error("give --port <serial port> or use --sim")

    buf = SampleBuffer()
    stop = threading.Event()
    target = sim_worker if args.sim else serial_worker
    wargs = (buf, stop) if args.sim else (args.port, args.baud, buf, stop)
    threading.Thread(target=target, args=wargs, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("pulse art")
    font = pygame.font.SysFont(None, 24)
    canvas = pygame.Surface(screen.get_size())
    palette = make_palette(N_POINTS)
    analyzer = Analyzer()
    clock = pygame.time.Clock()
    show_info = True
    show_signal = True
    started = time.monotonic()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_h:
                    show_info = not show_info
                elif e.key == pygame.K_s:
                    show_signal = not show_signal
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        if canvas.get_size() != screen.get_size():
            canvas = pygame.Surface(screen.get_size())

        now = time.monotonic()
        ts, vs = buf.snapshot()
        analysis = analyzer.update(ts, vs, now, dt)
        render(canvas, now, analysis, analyzer.energy, palette)
        screen.fill((0, 0, 0))
        apply_glow(screen, canvas)

        if show_signal:
            draw_signal_panel(screen, font, buf, ts, vs, analysis, analyzer, now)
        if show_info:
            status = f"{analyzer.bpm:.0f} bpm" if analyzer.has_signal and analyzer.bpm else (
                "waiting for pulse..." if not analyzer.has_signal else "locking on...")
            txt = f"{status}   dropped spikes: {buf.rejected}   fps: {clock.get_fps():.0f}"
            screen.blit(font.render(txt, True, (150, 150, 170)), (14, 12))
        pygame.display.flip()

        if args.max_seconds and now - started > args.max_seconds:
            if args.snapshot:
                pygame.image.save(screen, args.snapshot)
            running = False

    stop.set()
    pygame.quit()


if __name__ == "__main__":
    main()