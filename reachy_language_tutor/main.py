"""Reachy Mini voice conversation powered by ElevenLabs Agents.

Reachy's microphone streams to an ElevenLabs agent over a websocket; the agent's voice
plays through Reachy's speaker while the head wobbles with the speech, and the agent
calls client tools (nod, shake_head, celebrate, ...) to gesture.

The head camera keeps Reachy's attention on the person (see vision.py and motion.py).
When the agent calls `look`, or something socially notable happens (someone walks up,
waves), a camera frame is uploaded into the conversation so the LLM can see it.
"""

import asyncio
import difflib
import logging
import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp

from reachy_language_tutor.elevenlabs_agent import (
    GESTURE_TAGS,
    LANGUAGES,
    AgentManager,
    ConversationHandlers,
    SessionConfig,
    image_message,
    run_conversation,
)
from reachy_language_tutor.heart import HeartMonitor
from reachy_language_tutor.interaction import HeartDirector
from reachy_language_tutor.motion import GESTURES, MotionController
from reachy_language_tutor.vision import Vision

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_S = 0.1  # mic chunk sent to ElevenLabs
ECHO_TAIL_S = 0.4  # keep mic muted this long after Reachy stops talking (half-duplex)
# Barge-in gate: while Reachy talks, the mic only opens when it is this much louder than
# Reachy's own voice leaking back in, for BARGE_CHUNKS chunks in a row (~0.3 s).
BARGE_RATIO = 3.0
BARGE_FLOOR = 0.02
BARGE_CHUNKS = 3
REPEAT_SIMILARITY = 0.8  # a reply this close to the previous one, with no one speaking in between, is a loop
RECONNECT_TRIES = 3
CONTEXT_MIN_GAP_S = 20.0  # at most one camera note per kind this often


class StartRequest(BaseModel):
    language: str = "en"
    barge_in: bool = True  # let people interrupt Reachy (gated against its own echo)


class TextRequest(BaseModel):
    text: str


class GestureRequest(BaseModel):
    name: str


class PauseRequest(BaseModel):
    paused: bool


class SettingsRequest(BaseModel):
    follow: bool | None = None  # track faces / voices / movement with the head
    react: bool | None = None  # let what the camera sees trigger replies (greet, wave back)
    language: str | None = None  # language the heartbeat-started conversation uses
    barge_in: bool | None = None


class HeartSettings(BaseModel):
    source: str | None = None  # serial | fake | off
    drive: bool | None = None  # heartbeat wakes Reachy, starts and ends the conversation
    flow: str | None = None  # showcase (camera welcomes, heart joins later) | heart (heartbeat wakes Reachy)
    auto_talk: bool | None = None
    thump: bool | None = None
    mode: str | None = None  # mirror | entrain
    fake_bpm: float | None = None
    fake_finger: bool | None = None


# Seconds between proactive, camera-triggered replies of each kind.
COOLDOWNS = {"greet": 45.0, "wave": 20.0}


class Tutor(ConversationHandlers):
    """Owns one conversation at a time and bridges it to the robot."""

    def __init__(self, reachy: ReachyMini):
        self.reachy = reachy
        self.motion = MotionController(reachy)
        self.vision = Vision(reachy, self._on_vision_event)
        self.motion.vision = self.vision
        self.react = True
        self.language = "en"
        self._kickoff: str | None = None
        self._kickoff_camera = False  # send the first turn with a camera frame
        self.last_activity = time.monotonic()  # someone talking, Reachy talking, or a finger on the sensor
        self.last_user_speech = 0.0
        self.motion.on_voice = self._on_voice_wake
        self.heart = HeartMonitor(lambda k, i: self.director.on_heart_event(k, i),
                                  port=os.environ.get("HEART_PORT", ""),
                                  polarity=os.environ.get("HEART_POLARITY", "positive"))
        self.director = HeartDirector(self, os.environ.get("HEART_OSC", "127.0.0.1:10000"))
        self._conversation_id: str | None = None
        self._last_proactive: dict[str, float] = {}
        self.agents = AgentManager(
            api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", ""),
            agent_id=os.environ.get("ELEVENLABS_AGENT_ID", ""),
            llm=os.environ.get("ELEVENLABS_LLM", ""),
        )
        self.status = "idle"
        self.error = ""
        self.transcript: list[dict] = []
        self.config: SessionConfig | None = None
        self.barge_in = True
        self._speaking_until = 0.0
        self._echo_rms = 0.0
        self._loud_chunks = 0
        self._gate_open = False
        self._last_reply = ""
        self._heard_since_reply = True
        self._muted_reply = False
        self._last_context: dict[str, float] = {}
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._outgoing: asyncio.Queue | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def start_session(self, kickoff: str | None = None, camera: bool = False) -> None:
        """Start a conversation in the current language; `kickoff` is a first turn for the agent
        (sent with the current camera frame when `camera`)."""
        self.start(StartRequest(language=self.language, barge_in=self.barge_in), kickoff)
        self._kickoff_camera = camera
        self.last_activity = time.monotonic()

    # ---- dozing (showcase) ----

    def doze(self) -> None:
        """Nobody has talked for a while: end the conversation and doze, still watching."""
        if self.motion.dozing or self.motion.sleeping:
            return
        logger.info("Dozing: no interaction for a while")
        self.motion.dozing = True
        self.transcript.append({"role": "event", "text": "💤 Quiet for a while. Reachy is dozing (walk up, wave or say hi)"})
        threading.Thread(target=self.stop, daemon=True).start()

    def rouse(self, reason: str) -> bool:
        """Wake from dozing with a big hello. Returns False if Reachy wasn't dozing."""
        if not self.motion.dozing:
            return False
        logger.info("Waking from doze: %s", reason)
        self.motion.dozing = False
        self.last_activity = time.monotonic()
        self.motion.play("welcome")
        self.transcript.append({"role": "event", "text": f"☀️ Reachy woke up ({reason})"})
        return True

    def _welcome_visitors(self, hello: str) -> None:
        if self.status == "idle" and self.react and self.agents.api_key:
            try:
                self.start_session(kickoff=hello, camera=True)
            except Exception as e:
                logger.warning("Could not start the conversation: %s", e)

    def _on_voice_wake(self) -> None:
        if self.director.flow == "showcase" and self.rouse("heard a voice"):
            self._welcome_visitors(
                "(Someone talking near you just woke you from a little nap. This is what your camera sees. "
                "Greet them warmly and expressively in one or two short sentences and ask their name.)")

    def note(self, text: str, reply: bool) -> None:
        """A sensor note for the agent: a turn it answers (reply) or silent context."""
        if not self._connected():
            return
        if reply:
            self._new_input()
            msg = {"type": "user_message", "text": text}
        else:
            msg = {"type": "contextual_update", "text": text}
        self._loop.call_soon_threadsafe(self._outgoing.put_nowait, msg)

    def start(self, req: StartRequest, kickoff: str | None = None) -> None:
        # Re-read .env so a fixed key takes effect without restarting the app.
        load_dotenv(ENV_FILE, override=True)
        self.agents.api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.agents.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        self.agents.llm = os.environ.get("ELEVENLABS_LLM", "")
        if not self.agents.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set (put it in reachy_language_tutor/.env)")
        self.stop()
        self.wake()
        self.motion.dozing = False
        self.last_activity = time.monotonic()
        self.config = SessionConfig(req.language)
        self.language = req.language
        self._kickoff = kickoff
        self._stopping = False
        self.barge_in = req.barge_in
        # A heartbeat-started session keeps the heart events that led up to it.
        self.transcript = [m for m in self.transcript if m.get("heart")] if kickoff else []
        self.error = ""
        self.status = "connecting"
        self._thread = threading.Thread(target=self._run_session, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True  # an intended stop: don't reconnect
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def sleep(self) -> None:
        """End the conversation and put Reachy to sleep."""
        self.stop()
        self.vision.enabled = False
        self.motion.sleep()

    def wake(self) -> None:
        self.motion.dozing = False
        self.motion.wake()
        self.vision.enabled = not self.motion.paused

    def set_paused(self, paused: bool) -> None:
        """Freeze the body and turn camera analysis off; the conversation keeps going."""
        if paused:
            self.vision.enabled = False
            self.motion.pause()
            self._context("paused", "(Your camera and movement are switched off for now.)")
        else:
            self.motion.resume()
            self.vision.enabled = not self.motion.sleeping
            self._context("resumed", "(Your camera and movement are back on.)")

    def send_text(self, text: str) -> None:
        if self._loop and self._outgoing:
            self._new_input()
            self.transcript.append({"role": "user", "text": text})
            msg = {"type": "user_message", "text": text}
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, msg)

    def _run_session(self) -> None:
        self._loop = asyncio.new_event_loop()
        attempt = 0
        try:
            while True:
                try:
                    self._loop.run_until_complete(self._session())
                    break
                except Exception as e:
                    # The link to ElevenLabs can drop mid-visit (Wi-Fi blip, "connection reset",
                    # "no close frame"). Reconnect and carry on where we were, quietly.
                    if self._stopping or attempt >= RECONNECT_TRIES:
                        raise
                    attempt += 1
                    logger.warning("Conversation connection dropped (%s); reconnecting %d/%d",
                                   e, attempt, RECONNECT_TRIES)
                    self.status = "connecting"
                    self._conversation_id = None
                    self._speaking_until = 0.0
                    self.reachy.media.audio.clear_player()
                    if attempt == 1:
                        self.transcript.append({"role": "event", "text": "📶 Connection blipped, reconnecting…"})
                    self._kickoff, self._kickoff_camera = self._resume_note(), False
                    time.sleep(attempt)
                    if self._stopping:
                        break
        except Exception as e:
            logger.exception("Session failed")
            self.error = ("Lost the connection to ElevenLabs. Check the internet, then press Start talking "
                          f"or put a finger on the sensor to try again. ({e})")
        finally:
            self._loop.close()
            self._loop = None
            self.status = "idle"
            self._conversation_id = None
            self.motion.mode = "idle"
            self.reachy.media.audio.clear_player()

    def _resume_note(self) -> str:
        """First turn after a reconnect: the last few lines, so the agent picks up the thread."""
        lines = [f"{'Them' if m['role'] == 'user' else 'You'}: {m['text']}"
                 for m in self.transcript if m["role"] in ("user", "agent")][-6:]
        recap = " / ".join(lines) if lines else "you had just met"
        return ("(Your connection dropped for a moment and is back. The conversation so far: " + recap +
                ". Carry on naturally from there in one short sentence, in their language; don't repeat "
                "yourself and don't mention technical problems.)")

    async def _session(self) -> None:
        self._stop = asyncio.Event()
        self._outgoing = asyncio.Queue()
        self._mic_queue: asyncio.Queue = asyncio.Queue()
        url = await self.agents.prepare(self.config)
        mic_stop = threading.Event()
        mic_thread = threading.Thread(target=self._mic_reader, args=(self._loop, mic_stop), daemon=True)
        mic_thread.start()
        try:
            await run_conversation(url, self._mic_chunk, self, self._outgoing, self._stop)
        finally:
            mic_stop.set()
            mic_thread.join(timeout=1)

    # ---- audio ----

    def _mic_reader(self, loop: asyncio.AbstractEventLoop, stop: threading.Event) -> None:
        """Pull mic samples off the event loop: get_audio_sample() blocks for up to 20 ms."""
        while not stop.is_set():
            sample = self.reachy.media.get_audio_sample()
            if sample is not None:
                mono = sample.mean(axis=1) if sample.ndim == 2 else sample
                loop.call_soon_threadsafe(self._mic_queue.put_nowait, mono)

    async def _mic_chunk(self) -> bytes:
        """Collect ~CHUNK_S of mic audio as 16 kHz mono int16, muted while Reachy talks."""
        need = int(SAMPLE_RATE * CHUNK_S)
        parts, have = [], 0
        while have < need:
            mono = await self._mic_queue.get()
            parts.append(mono)
            have += len(mono)
        pcm = np.concatenate(parts)
        if self._speaking():
            if not self._barge_in_gate(pcm):
                pcm = np.zeros_like(pcm)
        else:
            self._loud_chunks = 0
            self._gate_open = False
            self.motion.mode = "listening"
        return (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()

    def _barge_in_gate(self, pcm: np.ndarray) -> bool:
        """While Reachy talks, pass the mic through only if someone is clearly talking over it.

        The mic also hears Reachy's own voice. Sending that would make the agent interrupt
        itself (or answer itself), so we learn how loud the echo is and open the gate only
        when the mic is several times louder than that for a moment.
        """
        if not self.barge_in:
            return False
        if self._gate_open:
            return True
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        if self._echo_rms == 0.0:
            self._echo_rms = rms
        if rms > max(BARGE_RATIO * self._echo_rms, BARGE_FLOOR):
            self._loud_chunks += 1
        else:
            self._loud_chunks = 0
            # Track the echo level slowly, from quiet-enough chunks only.
            self._echo_rms = 0.95 * self._echo_rms + 0.05 * rms
        if self._loud_chunks >= BARGE_CHUNKS:
            logger.info("Barge-in: mic %.3f vs echo %.3f", rms, self._echo_rms)
            self._gate_open = True
        return self._gate_open

    def on_agent_audio(self, pcm: np.ndarray) -> None:
        if self._muted_reply:
            return  # a repeated reply we cut off
        now = time.monotonic()
        self._speaking_until = max(now, self._speaking_until) + len(pcm) / SAMPLE_RATE
        self.motion.mode = "speaking"
        self.reachy.media.push_audio_sample(pcm)

    def on_interruption(self) -> None:
        self.reachy.media.audio.clear_player()
        self._speaking_until = 0.0
        self._heard_since_reply = True
        self._muted_reply = False
        self.motion.mode = "listening"

    # ---- events ----

    def on_user_transcript(self, text: str) -> None:
        if text.strip(" .…"):  # ElevenLabs sends "..." for silence
            self.last_activity = self.last_user_speech = time.monotonic()
            self.transcript.append({"role": "user", "text": text})
            self._new_input()

    def _new_input(self) -> None:
        """Someone (or the camera) gave the agent something new to answer."""
        self._heard_since_reply = True
        self._muted_reply = False

    def on_agent_response(self, text: str) -> None:
        tags = [t.strip().lower() for t in re.findall(r"\[([^\]]{1,30})\]", text)]
        # The tags are acted out by the voice, not spoken; strip them for the transcript.
        clean = re.sub(r"\[[^\]]{1,30}\]\s*", "", text).strip()
        if not clean.strip(" .…"):
            return
        if not self._heard_since_reply and self._is_repeat(clean):
            logger.info("Cut a repeated reply: %s", clean[:80])
            self._muted_reply = True
            self.reachy.media.audio.clear_player()
            self._speaking_until = 0.0
            return
        self._last_reply, self._heard_since_reply = clean, False
        for tag in tags:
            if tag in GESTURE_TAGS:
                logger.info("Gesture from [%s]: %s", tag, GESTURE_TAGS[tag])
                self.motion.play(GESTURE_TAGS[tag])
                break
        self.transcript.append({"role": "agent", "text": clean})

    def _is_repeat(self, text: str) -> bool:
        prev = self._last_reply
        if not prev:
            return False
        # Compare the opening, since a repeat often trails off or is cut differently.
        n = min(len(prev), len(text), 160)
        return difflib.SequenceMatcher(None, prev[:n], text[:n]).ratio() >= REPEAT_SIMILARITY

    def on_tool_call(self, name: str, params: dict) -> None:
        logger.info("Tool: %s", name)
        if name == "look" and self.motion.paused:
            self.transcript.append({"role": "event", "text": "📷 Camera is paused"})
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, {
                "type": "user_message",
                "text": "(Your camera is switched off right now, so you can't see anything.)",
            })
        elif name == "look":
            self._new_input()
            self._send_camera(
                "(This is what your camera sees right now. Mention only what is actually visible.)",
                "📷 Reachy looked",
            )
        elif name == "go_to_sleep":
            self.director.request_sleep()
        else:
            logger.warning("Unknown tool %s", name)

    def on_status(self, status: str) -> None:
        self.status = status

    def on_conversation_started(self, conversation_id: str) -> None:
        self._conversation_id = conversation_id
        if self._kickoff and self._kickoff_camera and self.vision.snapshot_jpeg() is not None:
            self.status = "connected"  # (the socket loop sets it right after this callback)
            text, self._kickoff, self._kickoff_camera = self._kickoff, None, False
            self._send_camera(text, "👋 Reachy welcomed someone")
        elif self._kickoff:
            self._new_input()
            msg = {"type": "user_message", "text": self._kickoff}
            self._kickoff = None
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, msg)

    # ---- seeing ----

    def _connected(self) -> bool:
        return self.status == "connected" and self._loop is not None and self._conversation_id is not None

    def _speaking(self) -> bool:
        return time.monotonic() < self._speaking_until + ECHO_TAIL_S

    def _send_camera(self, text: str, note: str) -> None:
        """Upload the current frame and send it as a turn the agent will answer."""
        jpeg = self.vision.snapshot_jpeg()
        if jpeg is None or not self._connected():
            return
        self.transcript.append({"role": "event", "text": note})
        self._new_input()
        loop, conv = self._loop, self._conversation_id

        async def go() -> None:
            try:
                file_id = await self.agents.upload_image(conv, jpeg)
                await self._outgoing.put(image_message(file_id, text))
            except Exception:
                logger.exception("Sending camera frame failed")

        asyncio.run_coroutine_threadsafe(go(), loop)

    def _context(self, kind: str, text: str) -> None:
        """Tell the agent something silently (no reply), sparingly.

        Each update is a turn on the agent's side, and a flood of them (a face flickering in
        and out of view) pushed the model to speak again. So: one per kind per
        CONTEXT_MIN_GAP_S, and never while Reachy is talking.
        """
        now = time.monotonic()
        if self._speaking() or now - self._last_context.get(kind, 0.0) < CONTEXT_MIN_GAP_S:
            return
        self._last_context[kind] = now
        if self._connected():
            msg = {"type": "contextual_update", "text": text}
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, msg)

    def _cooldown_ok(self, kind: str) -> bool:
        now = time.monotonic()
        if now - self._last_proactive.get(kind, 0.0) < COOLDOWNS[kind]:
            return False
        self._last_proactive[kind] = now
        return True

    def _on_vision_event(self, kind: str, info: dict) -> None:
        """Social events from the camera: react with the body, and tell the agent."""
        if self.motion.sleeping:
            return
        logger.info("Vision event: %s %s", kind, info)
        # While the heart interaction is wrapping up (finger gone), it owns the next turn.
        talk = (self.react and self._connected() and not self._speaking()
                and self.director.state not in ("searching", "goodbye"))
        showcase = self.director.flow == "showcase"
        if kind == "wave" and showcase and self.motion.dozing:
            kind = "engaged"  # a wave wakes a dozing Reachy, like walking up does
            self.rouse("someone waved")
        if kind == "engaged" and showcase:
            # A visitor stepped up: the big welcome, then the conversation (with what Reachy sees).
            if not self.rouse("someone walked up"):
                self.motion.play("welcome")
            self.last_activity = time.monotonic()
            group = info.get("faces", 1) > 1
            hello = ("(Visitors just walked up to you" if group else "(Someone just walked up to you") + (
                ". This is what your camera sees. Welcome them warmly and expressively in one or two "
                "short sentences, mention something you actually see, and ask their name.)")
            if self.status == "idle":
                self._welcome_visitors(hello)
            elif talk and self._cooldown_ok("greet"):
                self._send_camera(hello, "👋 Someone walked up")
            return
        if kind == "person_arrived" and showcase:
            self.motion.play("curious")  # noticed someone; the welcome waits until they step up
            return
        if kind == "person_arrived":
            if info["away_s"] > 20:
                self.motion.play("curious")
                if talk and self._cooldown_ok("greet"):
                    self._send_camera(
                        "(Someone just came up to you. This is what your camera sees. Greet them briefly and naturally.)",
                        "👋 Someone walked up",
                    )
            elif info["away_s"] > 8:
                self._context("presence", "(The person is back in front of you.)")
        elif kind == "person_left":
            if info["stayed_s"] > 4:
                self.transcript.append({"role": "event", "text": "🚶 The person walked out of view"})
                self._context("presence", "(The person walked out of your view.)")
        elif kind == "wave":
            self.motion.play("wave")
            if talk and self._cooldown_ok("wave"):
                self._send_camera("(The person is waving at you. This is your camera view.)", "👋 They waved")
        elif kind == "very_close":
            self.motion.play("surprised")

    def snapshot(self, since: int = 0) -> dict:
        return {
            "status": self.status,
            "mode": self.motion.mode,
            "sleeping": self.motion.sleeping,
            "dozing": self.motion.dozing,
            "paused": self.motion.paused,
            "error": self.error,
            "has_api_key": bool(self.agents.api_key),
            "config": asdict(self.config) if self.config else None,
            "vision": {
                "follow": self.motion.follow,
                "react": self.react,
                "attention": self.motion.attention,
                "faces": self.vision.percept.faces,
                "fps": round(self.vision.fps, 1),
            },
            "heart": self.heart_snapshot(signal=False),
            "transcript": self.transcript[since:],
            "transcript_len": len(self.transcript),
        }


    def heart_snapshot(self, signal: bool = True) -> dict:
        h = self.heart.snapshot()
        if not signal:
            h.pop("signal")
        d = self.director
        h.update({
            "state": d.state,
            "drive": d.drive,
            "flow": d.flow,
            "auto_talk": d.auto_talk,
            "thump": d.thump,
            "mode": self.motion.heart_mode,
            "speaking": self._speaking(),
            "talking": self.status == "connected",
            "sleeping": self.motion.sleeping,
        })
        return h


class ReachyLanguageTutor(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8043"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        tutor = Tutor(reachy_mini)
        self._register_routes(tutor)

        reachy_mini.media.start_recording()
        reachy_mini.media.start_playing()
        reachy_mini.media.enable_wobbling(tutor.motion.set_speech_offsets)
        tutor.motion.start()
        tutor.vision.start()
        tutor.heart.start(os.environ.get("HEART_SOURCE", "serial"))
        tutor.director.start()
        if tutor.director.flow == "heart":
            tutor.sleep()  # wait for the first heartbeat
        logger.info("Conversation app ready on %s", self.custom_app_url)
        try:
            stop_event.wait()
        finally:
            tutor.stop()
            tutor.director.stop()
            tutor.heart.stop()
            tutor.vision.stop()
            tutor.motion.stop()
            reachy_mini.media.disable_wobbling()
            reachy_mini.media.stop_playing()
            reachy_mini.media.stop_recording()

    def _register_routes(self, tutor: Tutor) -> None:
        from fastapi import HTTPException, Response
        from fastapi.responses import StreamingResponse

        api = self.settings_app
        assert api is not None

        @api.get("/api/options")
        def options() -> dict:
            return {"languages": LANGUAGES, "gestures": list(GESTURES)}

        @api.get("/api/state")
        def get_state(since: int = 0) -> dict:
            return tutor.snapshot(since)

        @api.post("/api/start")
        def start(req: StartRequest) -> dict:
            if req.language not in LANGUAGES:
                raise HTTPException(400, "Unknown language")
            try:
                tutor.start(req)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            return {"ok": True}

        @api.post("/api/stop")
        def stop() -> dict:
            tutor.stop()
            return {"ok": True}

        @api.post("/api/sleep")
        def sleep() -> dict:
            tutor.director.reset()
            tutor.sleep()
            return {"ok": True}

        @api.post("/api/wake")
        def wake() -> dict:
            tutor.wake()
            return {"ok": True}

        @api.post("/api/pause")
        def pause(req: PauseRequest) -> dict:
            tutor.set_paused(req.paused)
            return {"ok": True}

        @api.post("/api/settings")
        def settings(req: SettingsRequest) -> dict:
            if req.follow is not None:
                tutor.motion.follow = req.follow
            if req.react is not None:
                tutor.react = req.react
            if req.language in LANGUAGES:
                tutor.language = req.language
            if req.barge_in is not None:
                tutor.barge_in = req.barge_in
            return {"ok": True}

        @api.get("/api/camera.jpg")
        def camera_jpg() -> Response:
            jpeg = tutor.vision.preview_jpeg()
            if jpeg is None:
                raise HTTPException(503, "No camera frame yet")
            return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

        @api.get("/api/camera.mjpg")
        async def camera_mjpg() -> StreamingResponse:
            async def frames():
                while True:
                    jpeg = tutor.vision.preview_jpeg()
                    if jpeg:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

        @api.get("/api/heart")
        def heart() -> dict:
            return tutor.heart_snapshot()

        @api.post("/api/heart/settings")
        def heart_settings(req: HeartSettings) -> dict:
            d = tutor.director
            if req.source is not None:
                try:
                    tutor.heart.set_source(req.source)
                except ValueError:
                    raise HTTPException(400, "source must be serial, fake or off")
            if req.drive is not None:
                d.drive = req.drive
            if req.flow in ("showcase", "heart"):
                d.flow = req.flow
            if req.auto_talk is not None:
                d.auto_talk = req.auto_talk
            if req.thump is not None:
                d.thump = req.thump
            if req.mode in ("mirror", "entrain"):
                tutor.motion.heart_mode = req.mode
            tutor.heart.set_fake(req.fake_finger, req.fake_bpm)
            return tutor.heart_snapshot(signal=False)

        @api.get("/api/heart/raw")
        def heart_raw() -> dict:
            return {"lines": list(tutor.heart.raw_tail)}

        @api.post("/api/heart/beat")
        def heart_beat() -> dict:
            tutor.heart.manual_beat()
            return {"ok": True}

        @api.get("/api/heart/stream")
        async def heart_stream() -> StreamingResponse:
            """Server-sent events at ~30 Hz for the projected visuals (beats show as a counter)."""
            import json

            async def events():
                seq = -1
                while True:
                    snap = tutor.heart_snapshot(signal=False)
                    seq, raw = tutor.heart.samples_since(seq)
                    snap["raw"] = raw
                    snap["now"] = round(time.monotonic(), 4)
                    yield f"data: {json.dumps(snap)}\n\n"
                    await asyncio.sleep(0.033)

            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store"})

        @api.get("/visuals")
        def visuals() -> Response:
            """Victoria's pulse art, for the TV / projector."""
            html = (Path(__file__).parent / "static" / "pulse_art.html").read_text()
            return Response(html, media_type="text/html")

        @api.get("/visuals/ribbons")
        def visuals_ribbons() -> Response:
            html = (Path(__file__).parent / "static" / "visuals.html").read_text()
            return Response(html, media_type="text/html")

        @api.post("/api/text")
        def text(req: TextRequest) -> dict:
            tutor.send_text(req.text)
            return {"ok": True}

        @api.post("/api/gesture")
        def gesture(req: GestureRequest) -> dict:
            return {"ok": tutor.motion.play(req.name)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # The page polls several times a second; logging every request filled the disk over days.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    app = ReachyLanguageTutor()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
