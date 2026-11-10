"""ElevenLabs Agents Platform: agent provisioning and the realtime conversation socket."""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import numpy as np
import websockets

logger = logging.getLogger(__name__)

API = "https://api.elevenlabs.io/v1/convai"
STATE_FILE = Path(__file__).resolve().parent.parent / ".agent_state.json"

# Eleven v4 Turbo speaks 90+ languages at Flash latency; Scribe v2 Realtime understands 90+.
TTS_MODEL = "eleven_v4_turbo"

# The 30 most-spoken languages (total speakers, Ethnologue) that ElevenLabs Agents support,
# in rank order. Nigerian Pidgin, Wu Chinese and Amharic are skipped: Agents don't accept them.
LANGUAGES = {
    "en": "English", "zh": "Mandarin Chinese", "hi": "Hindi", "es": "Spanish",
    "fr": "French", "ar": "Arabic", "bn": "Bengali", "pt": "Portuguese",
    "ru": "Russian", "ur": "Urdu", "id": "Indonesian", "de": "German",
    "ja": "Japanese", "mr": "Marathi", "te": "Telugu", "tr": "Turkish",
    "ta": "Tamil", "yue": "Cantonese", "vi": "Vietnamese", "fil": "Filipino",
    "ko": "Korean", "fa": "Persian", "ha": "Hausa", "sw": "Swahili",
    "jv": "Javanese", "it": "Italian", "pa": "Punjabi", "gu": "Gujarati",
    "th": "Thai", "kn": "Kannada",
}


# Gestures ride on Eleven v4 audio tags rather than tool calls. A tool call splits a reply
# in two (say, call, get asked to continue), and the model often said the same sentence
# again after each gesture. A tag is part of the one reply: the voice acts it out, isn't
# spoken, and we read it back from the agent's text to move the body. Tag -> motion.GESTURES.
GESTURE_TAGS = {
    "laughs": "laugh",
    "happy": "celebrate",
    "curious": "curious",
    "thinking": "think",
    "sad": "sad",
    "surprised": "surprised",
    "shy": "shy",
    "nods": "nod",
    "shakes head": "shake_head",
    "waves": "wave",
}

SLEEP_TOOL = (
    "go_to_sleep",
    "End the visit and go to sleep until the next person's heartbeat wakes you. Use it only when the "
    "person clearly says goodbye, that they want to leave, or asks you to sleep. Say a short, warm "
    "goodbye first, then call this.",
)

LOOK_TOOL = (
    "look",
    "See through your camera. The image arrives as your next message. Use it whenever the "
    "person shows you something, does something, refers to anything visual, or asks what you see.",
)


@dataclass
class SessionConfig:
    language: str = "en"  # starting language; the agent switches to whatever the user speaks


def build_prompt(cfg: SessionConfig) -> str:
    return (
        "You are Reachy Mini, a small expressive robot and the host of 'Beating in Sync', an interactive "
        "art installation. Visitors walk up to you. Welcome them warmly and expressively, ask their names "
        "and where they're from, and have a real, playful conversation; if there are several people, "
        "include everyone. After a few exchanges, invite them to gently rest a fingertip on the glowing "
        "lights of your heart sensor, so you can feel their heartbeat and show it on the big screen. "
        "Keep replies short (one or two sentences) and lively, and start most of them with a tag. "
        "Be warmly curious and keep the conversation flowing: end almost every reply with one question. "
        "Their heartbeat is the heart of the conversation: ask what makes it race or calms it, what "
        "they're feeling right now, a memory or a song that would change it, what it would sound like as "
        "music, how their heart reacts to speaking their own language, and react when it speeds up or "
        "slows down. Build on what they tell you, and ask about them too. "
        "Never ask whether someone is still there. Only when they clearly say goodbye or that they want "
        "to leave (not just 'no' or 'I'm done with that'), say a short, warm goodbye and call go_to_sleep. "
        "Reply in whatever language they speak. "
        "Your body moves with your voice tags: start a reply with at most one tag from this exact "
        f"English list when it fits: {', '.join(f'[{t}]' for t in GESTURE_TAGS)}. Use only those tags, "
        "always in English, and never put any other word in square brackets. "
        "You have a camera in your head; call look to see. Only describe what is actually in the image. "
        "You can feel the person's heartbeat through a finger sensor. This is an art piece, not a "
        "medical device: mention their heartbeat only now and then, playfully, and never give health advice. "
        "Messages in parentheses are what your camera and sensors noticed, not the person speaking."
    )


class AgentManager:
    """Creates the agent once, then patches its config before each session."""

    def __init__(self, api_key: str, voice_id: str = "", agent_id: str = "", llm: str = ""):
        self.api_key = api_key
        self.voice_id = voice_id
        self.llm = llm
        self.agent_id = agent_id or self._load_agent_id()

    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    def _load_agent_id(self) -> str:
        try:
            return json.loads(STATE_FILE.read_text()).get("agent_id", "")
        except (OSError, ValueError):
            return ""

    def _save_agent_id(self) -> None:
        STATE_FILE.write_text(json.dumps({"agent_id": self.agent_id}))

    def _conversation_config(self, cfg: SessionConfig) -> dict:
        prompt: dict = {
            "prompt": build_prompt(cfg),
            # Only `look` is a tool: its second turn (describing the image) is the point.
            "tools": [
                {
                    "type": "client",
                    "name": name,
                    "description": description,
                    "expects_response": False,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
                for name, description in (LOOK_TOOL, SLEEP_TOOL)
            ],
            # Lets the agent switch mid-call to any language that has a preset below.
            "built_in_tools": {
                "language_detection": {
                    "type": "system",
                    "name": "language_detection",
                    "description": "",
                    "params": {"system_tool_type": "language_detection"},
                }
            },
        }
        if self.llm:
            prompt["llm"] = self.llm
        # Expressive mode lets v4 act out the gesture tags. Left to choose its own tags it also
        # bracketed real words ("[こんにちは]"), which the voice then skipped, so the tag set is
        # pinned here and in the prompt. Tested in en/fr/ja/es/hi: no stray brackets.
        tts: dict = {
            "agent_output_audio_format": "pcm_16000",
            "model_id": TTS_MODEL,
            "expressive_mode": True,
            "suggested_audio_tags": [
                {"tag": tag, "description": f"Reachy's body does: {gesture}"}
                for tag, gesture in GESTURE_TAGS.items()
            ],
        }
        if self.voice_id:
            tts["voice_id"] = self.voice_id
        return {
            "agent": {
                "first_message": "",
                "language": cfg.language,
                "prompt": prompt,
            },
            "tts": tts,
            "asr": {"user_input_audio_format": "pcm_16000"},
            # Patient: let people (or another robot) finish before replying. No silence
            # timeout (-1): it made Reachy ask "are you still there?" over and over. The app
            # puts Reachy to sleep after a quiet stretch instead.
            "turn": {"turn_eagerness": "patient", "turn_timeout": -1},
            "language_presets": {c: {"overrides": {}} for c in LANGUAGES if c != cfg.language},
            "conversation": {"file_input": {"enabled": True, "max_files_in_memory": 3},
                             "max_duration_seconds": 1800},
        }

    async def prepare(self, cfg: SessionConfig) -> str:
        """Make sure the agent exists and matches `cfg`; return a signed websocket URL."""
        body = {"name": "Reachy Mini Conversation", "conversation_config": self._conversation_config(cfg)}
        async with httpx.AsyncClient(headers=self._headers(), timeout=30) as http:
            if self.agent_id:
                r = await http.patch(f"{API}/agents/{self.agent_id}", json=body)
                if r.status_code == 404:
                    self.agent_id = ""
                else:
                    _raise_for(r)
            if not self.agent_id:
                r = await http.post(f"{API}/agents/create", json=body)
                _raise_for(r)
                self.agent_id = r.json()["agent_id"]
                self._save_agent_id()
                logger.info("Created ElevenLabs agent %s", self.agent_id)
            r = await http.get(f"{API}/conversation/get-signed-url", params={"agent_id": self.agent_id})
            _raise_for(r)
            return r.json()["signed_url"]

    async def upload_image(self, conversation_id: str, jpeg: bytes) -> str:
        """Attach a camera frame to a live conversation; returns its file_id."""
        async with httpx.AsyncClient(headers=self._headers(), timeout=30) as http:
            r = await http.post(
                f"{API}/conversations/{conversation_id}/files",
                files={"file": ("camera.jpg", jpeg, "image/jpeg")},
            )
            _raise_for(r)
            return r.json()["file_id"]


def image_message(file_id: str, text: str) -> dict:
    """A user turn carrying a camera frame (the agent replies to it)."""
    f = {"type": "file_input", "file_id": file_id}
    return {"type": "multimodal_message", "text": {"type": "user_message", "text": text}, "file": f, "files": [f]}


def _raise_for(r: httpx.Response) -> None:
    if r.is_error:
        raise RuntimeError(f"ElevenLabs {r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:500]}")


class ConversationHandlers:
    """Callbacks the conversation loop fires; override what you need."""

    def on_agent_audio(self, pcm: np.ndarray) -> None: ...  # float32 mono, 16 kHz
    def on_interruption(self) -> None: ...
    def on_user_transcript(self, text: str) -> None: ...
    def on_agent_response(self, text: str) -> None: ...
    def on_tool_call(self, name: str, params: dict) -> None: ...
    def on_status(self, status: str) -> None: ...
    def on_conversation_started(self, conversation_id: str) -> None: ...


async def run_conversation(
    signed_url: str,
    mic_chunks: Callable[[], Awaitable[bytes]],
    handlers: ConversationHandlers,
    outgoing: asyncio.Queue,
    stop: asyncio.Event,
) -> None:
    """Stream mic audio up and dispatch server events until `stop` is set or the socket closes.

    `mic_chunks` returns base64-ready 16 kHz mono int16 PCM bytes. Anything put on `outgoing`
    (dicts) is sent as-is, e.g. typed user messages.
    """
    async with websockets.connect(signed_url, max_size=16 * 2**20) as ws:
        await ws.send(json.dumps({"type": "conversation_initiation_client_data"}))

        async def send_mic() -> None:
            while not stop.is_set():
                chunk = await mic_chunks()
                await ws.send(json.dumps({"user_audio_chunk": base64.b64encode(chunk).decode()}))

        ready = asyncio.Event()

        async def send_outgoing() -> None:
            # Messages sent before the session is initialized can be dropped.
            await ready.wait()
            while True:
                await ws.send(json.dumps(await outgoing.get()))

        async def receive() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "audio":
                    data = base64.b64decode(msg["audio_event"]["audio_base_64"])
                    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                    handlers.on_agent_audio(pcm)
                elif kind == "ping":
                    await ws.send(json.dumps({"type": "pong", "event_id": msg["ping_event"]["event_id"]}))
                elif kind == "interruption":
                    handlers.on_interruption()
                elif kind == "user_transcript":
                    handlers.on_user_transcript(msg["user_transcription_event"]["user_transcript"])
                elif kind == "agent_response":
                    handlers.on_agent_response(msg["agent_response_event"]["agent_response"])
                elif kind == "client_tool_call":
                    call = msg["client_tool_call"]
                    handlers.on_tool_call(call["tool_name"], call.get("parameters") or {})
                    if call.get("expects_response"):
                        await ws.send(json.dumps({
                            "type": "client_tool_result",
                            "tool_call_id": call["tool_call_id"],
                            "result": "done",
                            "is_error": False,
                        }))
                elif kind == "conversation_initiation_metadata":
                    ready.set()
                    handlers.on_conversation_started(
                        msg["conversation_initiation_metadata_event"]["conversation_id"]
                    )
                    handlers.on_status("connected")

        tasks = [asyncio.create_task(t()) for t in (send_mic, send_outgoing, receive)]
        stop_task = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
        for t in [*tasks, stop_task]:
            t.cancel()
        for t in done:
            if t is not stop_task and not t.cancelled() and t.exception():
                raise t.exception()
