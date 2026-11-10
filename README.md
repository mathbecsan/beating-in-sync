---
title: Reachy Heartsong
emoji: 💓
colorFrom: pink
colorTo: green
sdk: static
pinned: false
short_description: Your heartbeat wakes Reachy Mini; it talks in your language and moves to your pulse
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Heartsong 💓

*Beating in Sync*, a CS334 interactive installation by Mathias Becerra and Victoria Virgen.

A visitor rests a fingertip on a glowing pulse sensor and their heartbeat wakes Reachy Mini. Its
antennas twitch with every beat, then it greets them, using what its camera sees, and talks with
them in their own language (30 of the world's most-spoken, switching as they do). Their pulse keeps
shaping how Reachy breathes and sways, and Victoria's *Pulse Art* turns it into flowing light on the
big screen. Reachy asks about what moves their heart, and when they say goodbye it goes back to
sleep until the next heartbeat.

Built on [ElevenLabs Agents](https://elevenlabs.io/docs/agents-platform) (speech-to-text, LLM and
expressive text-to-speech over one websocket), the Reachy Mini SDK, and an analog pulse sensor on an
Arduino.

## Setup

```bash
cp .env.example .env   # then set ELEVENLABS_API_KEY
pip install -e .
python -m reachy_language_tutor.main
```

Flash `arduino/heart_a0/heart_a0.ino` to the sensor board and plug it in over USB. Open
http://localhost:8043 for the control page (Pulse Art is in the Heart card; **Full screen on TV**
puts it on a second screen). Reachy sleeps until a heartbeat arrives. With no sensor, choose
*Fake heart* in the Heart card.

## Notes

- **Interrupting.** "Let me interrupt Reachy" (on by default) lets visitors talk over Reachy. The mic
  is gated against Reachy's own voice, so it never interrupts itself.
- **Languages.** The 30 most-spoken languages ElevenLabs Agents support; Reachy switches when the
  visitor does. Speech uses `eleven_v4_turbo`, recognition Scribe v2 Realtime.
- **Pick the language you'll speak.** It sets what speech recognition expects. With clear audio
  Reachy can follow you into another language, but in a noisy room switch the dropdown.
- **Expressive voice and gestures.** Reachy starts replies with one tag from a fixed list
  (`[laughs]`, `[happy]`, `[nods]`, ...). The voice acts it out and the app plays the matching
  body gesture. The list is fixed because, left to itself, the model bracketed real words.
- **Voice / LLM.** `ELEVENLABS_VOICE_ID` picks a voice, `ELEVENLABS_LLM` the model.
- **Connection drops** reconnect automatically and pick up the conversation where it was.

## Heartbeat interaction

**Visit flow (default: "Heartbeat starts each visit").** Reachy sleeps between visitors. A finger on
the sensor wakes it; its antennas sync to the pulse; once the pulse is steady Reachy greets the person
(using what its camera sees) and the conversation runs as long as they like. Lifting the finger
doesn't end it. A visit ends when the person says goodbye / they're finished / "go to sleep" (the
agent's `go_to_sleep` tool), or after 60 s with nobody talking; Reachy then sleeps until the next
heartbeat. Reachy never asks "are you still there?" (ElevenLabs turn timeout disabled). The other
flow, "Camera welcomes, heart joins later", greets people the camera sees walking up and dozes when quiet.
Pulse Art only shows a confirmed pulse; touches and noise leave it calm.

A participant's pulse brings Reachy to life. Placing a finger on the sensor wakes Reachy up.
For the first ~8 s every beat twitches its antennas, with a soft thump from the speaker.
Once the pulse is steady, the conversation starts on its own and Reachy greets them. The BPM
keeps shaping how Reachy breathes and sways: faster is livelier, slower is softer. Lifting
the finger makes Reachy look around; after 20 s it says goodbye and goes back to sleep for
the next person.

States: `waiting → waking → syncing → connected → searching → goodbye → waiting`
(see `interaction.py`).

**Our sensor (analog on A0).** Flash `arduino/heart_a0/heart_a0.ino` to the board. It just
prints `analogRead(A0)` 250 times a second; the laptop does everything else:
band-pass filtering, beat detection, BPM, and auto-flipping polarity if the pulse comes in upside down.
Mains hum and a floating pin are ignored. A finger counts as present after 5 beats in a steady
rhythm (about 3–6 s), and as gone when the rhythm falls apart or no beat comes for 3 s.
The board's LED flickering with your pulse is a quick wiring check.
Beats are the upward peaks (`HEART_POLARITY=positive`, the default, matches our circuit; use
`negative` if a different circuit puts the pulse upside down, or `auto`). Pressing or shifting
the finger drives the signal to its rails; those swings are ignored and don't count as the
finger leaving.

**Sensor board.** Plug the Arduino / ESP32 into the laptop over USB at 115200 baud. It is
picked up automatically (Reachy's own USB port is skipped); set `HEART_PORT` to force one.
Any of these line formats works:

| Line | Meaning |
| --- | --- |
| `BEAT` | one beat detected on the board |
| `BPM:72` | heart-rate estimate |
| `FINGER:1` / `FINGER:0` | finger placed / removed |
| `512` | a raw pulse sample (PulseSensor `analogRead`); beats are detected on the laptop |
| `IR=51234, BPM=71, ...` | MAX30102 (SparkFun example) output; IR under 50000 = no finger |

**Without the sensor**, pick *Fake heart* in the Heart card: tick *Finger on sensor* and set a BPM.
The space bar taps a beat by hand. Use this for development and as a demo backup.

**TV / projection.** Open `http://localhost:8043/visuals` on the TV's screen and press F (or
double-click) for full screen. This is Victoria's *Pulse Art* (`victoria/pulse_art.py`), ported to
the browser and fed by the app's sensor stream instead of opening the serial port itself (only one
program can own the port, and the app needs it to drive Reachy). Her ribbons trace the raw A0
pulse; the glow flares on the same beats Reachy moves to. Keys: S her signal-debug panel,
H info line, C caption. The earlier ribbon visual is at `/visuals/ribbons`. The same events go out over OSC to `HEART_OSC` for
TouchDesigner: `/heart/beat bpm`, `/heart/bpm bpm` (once steady), `/heart/finger 0|1`,
`/heart/live bpm finger` (5 Hz), `/reachy/state name`.

**Rhythm.** *Mirror* breathes at the participant's pace; *Breathe slower (entrain)* breathes
20% slower, to study whether their heart rate drifts down.

Heart data is only kept in memory and shown live; nothing is saved. This is an art piece, not a medical reading.
