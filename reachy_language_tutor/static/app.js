const $ = (id) => document.getElementById(id);
let seen = 0;
let running = false;

const MODES = {
  idle: "Reachy is resting",
  listening: "🎧 Reachy is listening, go ahead",
  speaking: "🗣️ Reachy is speaking",
};

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function init() {
  const opts = await api("/api/options");
  const langs = Object.entries(opts.languages).sort((a, b) => a[1].localeCompare(b[1]));
  for (const [code, name] of langs) $("language").add(new Option(name, code));
  $("language").value = "en";
  for (const g of opts.gestures) {
    const b = document.createElement("button");
    b.textContent = g.replace("_", " ");
    b.onclick = () => api("/api/gesture", { name: g });
    $("gestures").append(b);
  }
  try {
    const saved = JSON.parse(localStorage.getItem("tutorSetup") || "{}");
    if (saved.language) $("language").value = saved.language;
  } catch {}
  syncSession();
  poll();
}

// Recognition is tuned to the chosen language, so switching mid-conversation restarts it.
$("language").onchange = () => { if (running) $("startBtn").click(); };

$("startBtn").onclick = async () => {
  const body = { language: $("language").value, barge_in: $("bargeIn").checked };
  try { localStorage.setItem("tutorSetup", JSON.stringify({ language: body.language })); } catch {}
  $("error").hidden = true;
  try { await api("/api/start", body); seen = 0; $("transcript").innerHTML = ""; }
  catch (e) { showError(e.message); }
};

$("stopBtn").onclick = () => api("/api/stop", {});

let paused = false;
let pauseBusy = false;
const PAUSE_ICON = $("pauseBtn").querySelector("svg").outerHTML;
const PLAY_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" fill="currentColor"/></svg>';

function renderPauseBtn(label) {
  const b = $("pauseBtn");
  b.disabled = pauseBusy || sleeping;
  b.classList.toggle("resume", paused);
  b.innerHTML = (paused ? PLAY_ICON : PAUSE_ICON) + `<span>${label || (paused ? "Resume motion & camera" : "Pause motion & camera")}</span>`;
}

$("pauseBtn").onclick = async () => {
  pauseBusy = true;
  renderPauseBtn(paused ? "Resuming…" : "Pausing…");
  try { await api("/api/pause", { paused: !paused }); }
  catch (e) { showError(e.message); }
  finally { pauseBusy = false; }
};

$("follow").onchange = () => api("/api/settings", { follow: $("follow").checked });
$("react").onchange = () => api("/api/settings", { react: $("react").checked });

const ATTENTION = {
  face: "👀 looking at you",
  voice: "👂 turning to a voice",
  motion: "↪ something moved",
  idle: "looking around",
};

function renderVision(v, asleep) {
  if (!v) return;
  const off = asleep || paused;
  const a = off ? "idle" : v.attention;
  $("attn").textContent = asleep ? "💤 asleep" : paused ? "⏸ paused" : (v.faces > 1 && a === "face" ? `👀 ${v.faces} people, looking at one` : ATTENTION[a] || a);
  $("attn").className = "badge " + (asleep ? "" : paused ? "paused" : a);
  $("fps").textContent = off ? "camera off" : v.fps ? `${v.fps} fps` : "camera starting…";
  $("camOff").hidden = !off;
  $("camOffText").textContent = asleep ? "Camera off while Reachy sleeps" : "Camera and motion paused";
  $("follow").disabled = $("react").disabled = off;
  if (document.activeElement !== $("follow")) $("follow").checked = v.follow;
  if (document.activeElement !== $("react")) $("react").checked = v.react;
}

let sleeping = false;
let sleepBusy = false;
const MOON = $("sleepBtn").querySelector("svg").outerHTML;
const SUN = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="12" cy="12" r="5" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1.5v2.5M12 20v2.5M1.5 12h2.5M20 12h2.5M4.6 4.6l1.8 1.8M17.6 17.6l1.8 1.8M4.6 19.4l1.8-1.8M17.6 6.4l1.8-1.8"/></g></svg>';

function renderSleepBtn(label) {
  const b = $("sleepBtn");
  b.disabled = sleepBusy;
  b.classList.toggle("waking", sleeping);
  b.innerHTML = (sleeping ? SUN : MOON) + `<span>${label || (sleeping ? "Wake up" : "Stop & sleep")}</span>`;
}

$("sleepBtn").onclick = async () => {
  sleepBusy = true;
  renderSleepBtn(sleeping ? "Waking up…" : "Going to sleep…");
  try { await api(sleeping ? "/api/wake" : "/api/sleep", {}); }
  catch (e) { showError(e.message); }
  finally { sleepBusy = false; }
};

$("textForm").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("textInput").value.trim();
  if (!text) return;
  $("textInput").value = "";
  await api("/api/text", { text });
};

function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

async function poll() {
  try {
    const s = await api(`/api/state?since=${seen}`);
    running = s.status !== "idle";
    sleeping = s.sleeping;
    paused = s.paused;
    const pill = sleeping && !running ? "sleeping" : s.status;
    $("status").textContent = pill;
    $("status").className = "pill " + pill;
    if (!sleepBusy) renderSleepBtn();
    if (!pauseBusy) renderPauseBtn();
    renderVision(s.vision, sleeping);
    $("gestures").querySelectorAll("button").forEach((b) => (b.disabled = sleeping || paused));
    $("keyWarning").hidden = s.has_api_key;
    $("startBtn").disabled = !s.has_api_key;
    $("stopBtn").disabled = !running;
    $("textInput").disabled = $("sendBtn").disabled = s.status !== "connected";
    $("mode").textContent = s.dozing ? "💤 Reachy is dozing. Walk up, wave or say hi to wake it."
      : running ? (MODES[s.mode] || s.mode)
      : sleeping ? "💤 Reachy is asleep. Press Start talking or Wake up." : MODES.idle;
    showError(s.error || "");  // clears once the app recovers
    if (s.transcript_len < seen) { seen = 0; $("transcript").innerHTML = ""; }
    for (const m of s.transcript) {
      $("transcript").querySelector(".empty")?.remove();
      const p = document.createElement("p");
      p.className = m.role;
      p.textContent = m.text;
      $("transcript").append(p);
      $("transcript").scrollTop = $("transcript").scrollHeight;
    }
    seen += s.transcript.length;
  } catch {}
  setTimeout(poll, 400);
}

// ---- heart ----
// Victoria's Pulse Art lives in the Heart card; this makes it fill the screen (the TV).
$("artFull").onclick = () => {
  const el = $("artFrame");
  (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
};
const HEART_STATES = {
  waiting: "asleep, waiting for a heartbeat", waking: "waking up", syncing: "syncing to the pulse",
  connected: "visiting", searching: "finger lifted", goodbye: "saying goodbye", awake: "awake, waiting for a finger",
};
const SHOWCASE_STATES = { waiting: "waiting for a finger", waking: "waking up", syncing: "feeling the pulse", connected: "in sync" };
const heartApi = (body) => api("/api/heart/settings", body).catch((e) => showError(e.message));
$("hSource").onchange = () => heartApi({ source: $("hSource").value });
$("hMode").onchange = () => heartApi({ mode: $("hMode").value });
$("hFlow").onchange = () => heartApi({ flow: $("hFlow").value });
$("hThump").onchange = () => heartApi({ thump: $("hThump").checked });
$("fFinger").onchange = () => heartApi({ fake_finger: $("fFinger").checked });
$("fBpm").oninput = () => { $("fBpmOut").textContent = $("fBpm").value; };
$("fBpm").onchange = () => heartApi({ fake_bpm: Number($("fBpm").value) });
// The conversation the heartbeat starts uses the page's language and barge-in choice.
const syncSession = () => api("/api/settings", { language: $("language").value, barge_in: $("bargeIn").checked });
$("language").addEventListener("change", syncSession);
$("bargeIn").addEventListener("change", syncSession);
addEventListener("keydown", (e) => {
  if (e.code !== "Space" || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement.tagName)) return;
  e.preventDefault();
  api("/api/heart/beat", {});
});

let heartBeats = null;
function renderHeart(h) {
  if (!h) return;
  const on = h.finger;
  $("heartIcon").classList.toggle("off", !on);
  if (heartBeats !== null && h.beats > heartBeats && on) {
    $("heartIcon").classList.add("beat");
    setTimeout(() => $("heartIcon").classList.remove("beat"), 130);
  }
  heartBeats = h.beats;
  $("heartBpm").textContent = on && h.bpm ? Math.round(h.bpm) : "--";
  $("heartState").textContent = h.flow === "showcase" ? (SHOWCASE_STATES[h.state] || h.state)
    : h.drive ? (HEART_STATES[h.state] || h.state) : "not driving Reachy";
  $("heartState").className = "pill " + (h.state === "connected" ? "heart-connected" : h.state);
  $("heartSource").textContent = h.source === "serial"
    ? (h.port ? `Sensor connected on ${h.port}` : (h.port_error || "Looking for the sensor…"))
    : h.source === "fake" ? "Using a fake heart. Tick “Finger on sensor” to simulate a participant." : "Heart sensor off.";
  $("fakeControls").hidden = h.source !== "fake";
  $("sensorWarning").hidden = !(h.source === "serial" && !h.port);
  const keep = (id, v, prop = "value") => { if (document.activeElement !== $(id)) $(id)[prop] = v; };
  keep("hSource", h.source); keep("hMode", h.mode);
  keep("hFlow", h.flow); keep("hThump", h.thump, "checked");
  keep("fFinger", h.fake_finger, "checked");
  if (document.activeElement !== $("fBpm")) { $("fBpm").value = h.fake_bpm; $("fBpmOut").textContent = Math.round(h.fake_bpm); }
}

// The waveform is fetched faster than the main state, so the trace looks continuous.
async function pollWave() {
  try {
    const h = await api("/api/heart");
    renderHeart(h);
    const c = $("heartWave");
    if (!c) throw 0;
    const g = c.getContext("2d");
    c.width = c.clientWidth * (devicePixelRatio || 1); c.height = 60 * (devicePixelRatio || 1);
    g.clearRect(0, 0, c.width, c.height);
    const sig = h.signal || [];
    if (sig.length > 1 && h.finger) {
      g.beginPath();
      sig.forEach((v, i) => {
        const x = (i / (sig.length - 1)) * c.width;
        const y = c.height * (0.6 - 0.35 * Math.max(-1.2, Math.min(1.5, v)));
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.strokeStyle = "#e0356b"; g.lineWidth = 2 * (devicePixelRatio || 1); g.stroke();
    }
  } catch {}
  setTimeout(pollWave, 100);
}

init();
pollWave();
