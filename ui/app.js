"use strict";

const state = {
  detections: [], memory: [], safePath: null, safety: null, safetyTimer: null,
  goal: null, finding: false, findTarget: "", focused: null, focusedTimer: null,
  findTimer: null, socketOpen: false, showAll: false, lastMessage: null, recognition: null,
  lastSpoken: "", lastSpokenAt: 0, lastGoalCommand: "", targetState: null,
  queryTimer: null, searchOutcome: null, refreshInFlight: false,
};
const $ = id => document.getElementById(id);
const escapeHtml = (value = "") => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const labelKey = value => String(value || "object").trim().toLowerCase();
const titleCase = value => String(value || "object").replace(/\b\w/g, char => char.toUpperCase());
const numeric = value => Number.isFinite(Number(value)) ? Number(value) : null;
const distanceText = value => { const metres = numeric(value); return metres === null ? "Distance unavailable" : metres < 1 ? `${Math.round(metres * 100)} cm` : `${metres.toFixed(1)} m`; };
const directionText = value => titleCase(String(value || "position unknown").replaceAll("-", " "));
const clockDirection = clock => { const value = String(clock || ""); if (value.includes("12")) return "Ahead"; if (value.includes("6")) return "Behind"; if (/\b(1|2|3|4|5)\b/.test(value)) return "Right"; if (/\b(7|8|9|10|11)\b/.test(value)) return "Left"; return value || "Position unknown"; };
const relativeTime = timestamp => {
  if (!timestamp) return "not recently confirmed";
  if (typeof timestamp === "string" && !/^\d+(\.\d+)?$/.test(timestamp)) return timestamp;
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - Number(timestamp)));
  if (seconds < 2) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
};

function cleanAssistantText(value = "") {
  let text = String(value);
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  text = text.replace(/<think>[\s\S]*$/gi, "");
  return text.trim();
}

function currentTarget() {
  return labelKey(state.targetState?.label || state.goal?.target || state.goal?.label || state.searchOutcome?.target || state.findTarget);
}

function normaliseRequestedObject(value) {
  let target = String(value || "").toLowerCase().trim()
    .replace(/[?!.]+$/g, "")
    .replace(/^(?:where (?:is|are)|find|locate|show me|help me find|take me to|guide me to|navigate me to)\s+/i, "")
    .replace(/^(?:my|the|a|an)\s+/i, "")
    .replace(/\s+(?:please|for me)$/i, "").trim();
  const aliases = {"photo frame":"picture frame", "photo picture":"picture frame", "wall picture":"picture frame", "t shirt":"shirt", "t-shirt":"shirt", "key":"keys", "keychain":"keys", "phone":"cell phone", "mobile":"cell phone", "eye glasses":"eyeglasses", "glasses":"eyeglasses", "spectacles":"eyeglasses"};
  return aliases[target] || target;
}

function requestedObjectFromQuery(text) {
  const value = String(text || "").toLowerCase();
  const match = value.match(/(?:where\s+(?:is|are)|find|locate|show me|help me find|take me to|guide me to|navigate me to)\s+(.+)/i);
  return match ? normaliseRequestedObject(match[1]) : "";
}

function activateFindMode(target) {
  const label = normaliseRequestedObject(target);
  if (!label) return;
  clearTimeout(state.findTimer);
  state.findTarget = label;
  state.finding = !state.goal;
  state.searchOutcome = null;
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:"set_open_vocab", classes:[label]}));
  state.findTimer = setTimeout(() => {
    if (!state.goal) {
      state.finding = false; state.findTarget = "";
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:"set_open_vocab", classes:[]}));
      renderAll();
    }
  // A cold focused model can take longer than the initial camera scan.  Keep
  // the requested target active long enough for it to return a real result.
  }, 45000);
  renderAll();
}

function uiMode() {
  const status = state.goal?.status;
  if (status && !["idle", "cancelled", "not_found"].includes(status)) return "navigate";
  if (state.finding) return "find";
  return "explore";
}

function renderMode() {
  const mode = uiMode();
  document.body.dataset.mode = mode;
  $("modeBadge").textContent = mode.toUpperCase();
}

function rankedDetections() {
  const grouped = new Map();
  state.detections.forEach(item => {
    const key = labelKey(item.label);
    const previous = grouped.get(key);
    const itemDistance = numeric(item.distance_m) ?? 99;
    const previousDistance = numeric(previous?.distance_m) ?? 99;
    if (!previous || itemDistance < previousDistance || (itemDistance === previousDistance && Number(item.confidence || 0) > Number(previous.confidence || 0))) grouped.set(key, item);
  });
  return [...grouped.values()].sort((a, b) => (numeric(a.distance_m) ?? 99) - (numeric(b.distance_m) ?? 99));
}

function isHazard(item) {
  const alertMatch = state.safety && labelKey(state.safety.label) === labelKey(item.label);
  return Boolean(alertMatch && ["critical", "warning"].includes(state.safety.level));
}

function renderNearby() {
  const target = currentTarget();
  let items = rankedDetections();
  items.sort((a, b) => Number(labelKey(b.label) === target) - Number(labelKey(a.label) === target) || Number(isHazard(b)) - Number(isHazard(a)) || (numeric(a.distance_m) ?? 99) - (numeric(b.distance_m) ?? 99));
  const limit = state.showAll ? items.length : 4;
  $("nearbyObjects").innerHTML = items.length ? items.slice(0, limit).map(item => {
    const targetClass = labelKey(item.label) === target ? " target" : "";
    const hazardClass = isHazard(item) ? " hazard" : "";
    const direction = item.direction ? directionText(item.direction) : clockDirection(item.clock_direction);
    return `<article class="object-row${targetClass}${hazardClass}"><i aria-hidden="true"></i><div><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(direction)}</small></div><span>${escapeHtml(distanceText(item.distance_m))}</span></article>`;
  }).join("") : '<p class="empty">No reliable objects are visible.</p>';
  $("viewAll").hidden = items.length <= 4;
  $("viewAll").textContent = state.showAll ? "Show less" : "View all";
}

function normalizeGoal(raw) {
  if (!raw || raw.status === "idle") return null;
  if (raw.target) return raw;
  return {
    ...raw, target: raw.label, distance_m: raw.smoothed_distance,
    heading_error_deg: raw.smoothed_heading_error, command: raw.last_command,
  };
}

// All target-facing components read this one derived object. The backend goal
// distance is world/pose based; a detection bbox is used only for drawing.
function canonicalTarget() {
  const goal = state.goal;
  if (!goal) return null;
  return {
    label: goal.target || goal.label,
    distance: numeric(goal.distance_m ?? goal.smoothed_distance),
    direction: goal.direction,
    bearing: numeric(goal.heading_error_deg ?? goal.smoothed_heading_error),
    visible: Boolean(goal.visible || goal.tracking_state === "visible"),
    lastSeen: goal.last_visual_confirmation,
    status: goal.status,
    tracking: goal.tracking_state,
  };
}

function goalViewState() {
  if (state.safety?.level === "critical" && state.goal) return "obstacle";
  const status = state.goal?.status;
  if (status === "complete") return "arrived";
  if (["blocked"].includes(status)) return "obstacle";
  if (["lost", "unreliable", "not_found"].includes(status)) return "lost";
  if (status === "active") return "navigating";
  if (state.finding) return "finding";
  if (state.searchOutcome?.status === "found") return "located";
  if (state.searchOutcome?.status === "not_found") return "not-found";
  return "idle";
}

function renderGoal() {
  const view = goalViewState();
  const goal = state.goal;
  const active = canonicalTarget();
  const outcome = state.searchOutcome;
  const target = active?.label || outcome?.target || state.findTarget;
  const panel = $("goalPanel");
  panel.className = `goal-panel goal--${view}`;
  const stateLabels = {idle:"Ready", finding:"Searching", navigating:"On the way", obstacle:"Path blocked", arrived:"Arrived", lost:"Target lost", located:"Located", "not-found":"Not found"};
  $("goalState").innerHTML = `<i aria-hidden="true"></i>${stateLabels[view]}`;
  $("goalContext").textContent = ({idle:"Standing by", finding:"Looking through camera + memory", navigating:"Live turn-by-turn guidance", obstacle:"Safety pause", arrived:"Journey complete", lost:"Visual lock interrupted", located:"Latest search result", "not-found":"Search complete"})[view];
  $("goalHeading").textContent = view === "idle" ? "Ready when you are" : view === "arrived" ? `${titleCase(target)} reached` : view === "lost" ? `Find ${titleCase(target)}` : view === "located" ? `${titleCase(target)} located` : view === "not-found" ? `${titleCase(target)} not in view` : `${view === "finding" ? "Find" : "Reach"} ${titleCase(target)}`;
  const metres = active?.distance ?? null;
  $("goalDistance").textContent = metres !== null ? `${distanceText(metres)} remaining` : view === "located" ? `${distanceText(outcome?.distance)} · ${directionText(outcome?.direction)}` : view === "not-found" ? "No reliable live or remembered location was found." : view === "finding" ? "Checking the current view and recent memory…" : view === "idle" ? "Ask DrishtiSense to find or guide you to an object." : "Waiting for a reliable position.";
  const progress = ({idle:0, finding:38, navigating:68, obstacle:68, arrived:100, lost:68, located:100, "not-found":100})[view];
  $("goalProgress").style.width = `${progress}%`;
  panel.classList.toggle("goal--progressing", view === "finding");
  const facts = [];
  if (view === "navigating" || view === "obstacle" || view === "arrived") {
    facts.push(["Target state", active?.visible ? "Visually locked" : "Reacquiring from remembered position"]);
    facts.push(["Direction", directionText(active?.direction || "updating")]);
    facts.push(["Last visual confirmation", relativeTime(active?.lastSeen)]);
  } else if (view === "lost") {
    facts.push(["Next step", "Stop and slowly scan around"]);
  } else if (view === "located") {
    facts.push(["Direction", directionText(outcome?.direction || "nearby")]);
    facts.push(["Source", outcome?.visible === false ? "Remembered location" : "Live camera"]);
  } else if (view === "not-found") {
    facts.push(["Try next", "Pan slowly from left to right, then ask again"]);
  }
  $("goalFacts").innerHTML = facts.map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  $("cancelGoal").hidden = !goal || !["active", "blocked"].includes(goal.status);
  renderMode();
}

function fallbackGoalHud(goal) {
  if (!goal) return null;
  if (goal.status === "complete") return "STOP · TARGET AHEAD";
  if (goal.status === "blocked") return "STOP · PATH BLOCKED";
  const distance = numeric(goal.distance_m ?? goal.smoothed_distance);
  const suffix = distance === null ? "" : ` · ${distance.toFixed(1)}m`;
  const command = goal.command || goal.last_command;
  if (command?.includes("around")) return "↻ TURN AROUND";
  if (command?.includes("right")) return `${command.includes("slight") || command.includes("avoid") ? "↗ SLIGHTLY RIGHT" : "↻ TURN RIGHT"}${suffix}`;
  if (command?.includes("left")) return `${command.includes("slight") || command.includes("avoid") ? "↖ SLIGHTLY LEFT" : "↺ TURN LEFT"}${suffix}`;
  return `↑ WALK STRAIGHT${suffix}`;
}

function renderGuidance() {
  const hud = $("guidanceHud");
  const main = $("guidanceMain");
  const detail = $("guidanceDetail");
  const icon = $("guidanceIcon");
  const critical = state.safety?.level === "critical" || state.safePath?.status === "blocked";
  if (critical) {
    const label = state.safety?.label || "obstacle";
    const metres = state.safety?.distance_m;
    hud.className = "guidance guidance--critical"; icon.textContent = "!"; main.textContent = "STOP · OBSTACLE AHEAD";
    const detour = state.safePath?.direction && state.safePath.direction !== "center" ? ` · Move ${state.safePath.direction}` : "";
    detail.textContent = `${titleCase(label)}${metres == null ? "" : ` · ${distanceText(metres)}`}${detour}`;
    return;
  }
  if (state.safety?.level === "warning") {
    const label = state.safety.label || "obstacle";
    const metres = state.safety.distance_m;
    hud.className = "guidance guidance--warning"; icon.textContent = "!"; main.textContent = "CAUTION · OBSTACLE NEARBY";
    const detour = state.safePath?.direction && state.safePath.direction !== "center" ? ` · Move ${state.safePath.direction}` : " · Continue carefully";
    detail.textContent = `${titleCase(label)}${metres == null ? "" : ` · ${distanceText(metres)}`}${detour}`;
    return;
  }
  if (state.goal) {
    const status = state.goal.status;
    const text = state.goal.hud || fallbackGoalHud(state.goal);
    const active = canonicalTarget();
    if (status === "complete") { const safeStop=String(state.goal.hud||"").includes("STOP HERE"); hud.className = "guidance guidance--arrived"; icon.textContent = "◎"; main.textContent = safeStop ? "STOP HERE · TARGET AHEAD" : "YOU'RE HERE"; detail.textContent = `${titleCase(active?.label)} directly ahead · ${distanceText(active?.distance)}`; return; }
    if (["lost", "unreliable", "not_found"].includes(status)) { hud.className = "guidance guidance--critical"; icon.textContent = "?"; main.textContent = "TARGET LOST"; detail.textContent = "Stop and scan around slowly"; return; }
    if (!active?.visible) {
      hud.className = "guidance guidance--warning"; icon.textContent = "↶"; main.textContent = "TARGET LOST";
      detail.textContent = `Look ${directionText(active?.direction || "around").toLowerCase()} · last seen ${relativeTime(active?.lastSeen)}`;
      return;
    }
    if (text) {
      const leading = text.trim().match(/^[↑↗↖↻↺←→■]/)?.[0] || "↑";
      hud.className = `guidance ${status === "blocked" ? "guidance--critical" : ""}`; icon.textContent = leading; main.textContent = text.replace(/^[↑↗↖↻↺←→■]\s*/, "");
      detail.textContent = `${titleCase(active?.label)} · ${distanceText(active?.distance)}${active?.visible ? " · target visible" : " · reacquiring"}`;
      return;
    }
  }
  if (state.finding) { hud.className = "guidance guidance--warning"; icon.textContent = "◎"; main.textContent = `FINDING ${titleCase(state.findTarget).toUpperCase()}`; detail.textContent = "Move the camera slowly across the room"; return; }
  const path = state.safePath;
  if (!path) { hud.className = "guidance guidance--waiting"; icon.textContent = "•"; main.textContent = "ASSESSING SURROUNDINGS"; detail.textContent = "Hold the camera steady"; return; }
  const direction = path.direction || "center";
  const metres = numeric(path.clearance_m);
  hud.className = `guidance ${path.status === "uncertain" ? "guidance--warning" : ""}`;
  icon.textContent = direction === "left" ? "←" : direction === "right" ? "→" : "↑";
  main.textContent = direction === "left" ? "OPEN SPACE LEFT" : direction === "right" ? "OPEN SPACE RIGHT" : "PATH AHEAD CLEAR";
  detail.textContent = metres === null ? "Continue with care" : `Approximately ${metres.toFixed(1)} m clear`;
}

function memoryCoordinates(item) {
  const relative = item.relative_coordinates || {};
  return {
    x: numeric(item.translation_x ?? relative.x),
    z: numeric(item.translation_z ?? relative.z),
  };
}

function usefulMemories() {
  const visibleLabels = new Set(rankedDetections().map(item => labelKey(item.label)));
  const target = currentTarget();
  const byLabel = new Map();
  state.memory.forEach(item => {
    const key = labelKey(item.object || item.label);
    const previous = byLabel.get(key);
    const candidateDistance = numeric(item.distance ?? item.distance_m) ?? 99;
    const previousDistance = numeric(previous?.distance ?? previous?.distance_m) ?? 99;
    if (!previous || item.visible || candidateDistance < previousDistance) byLabel.set(key, item);
  });
  return [...byLabel.values()].sort((a, b) => Number(labelKey(b.object || b.label) === target) - Number(labelKey(a.object || a.label) === target) || Number(Boolean(b.visible)) - Number(Boolean(a.visible)) || (numeric(a.distance ?? a.distance_m) ?? 99) - (numeric(b.distance ?? b.distance_m) ?? 99)).filter(item => labelKey(item.object || item.label) === target || !visibleLabels.has(labelKey(item.object || item.label)));
}

function renderMemory() {
  const memories = usefulMemories().slice(0, 3);
  $("memoryList").innerHTML = memories.length ? memories.map(item => {
    const label = item.object || item.label;
    const direction = directionText(item.direction);
    const lastSeen = item.time_ago || relativeTime(item.last_seen_timestamp || item.last_seen);
    return `<article class="memory-row"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(direction)} · ${escapeHtml(distanceText(item.distance ?? item.distance_m))}</small><small>Last seen ${escapeHtml(lastSeen)}</small></article>`;
  }).join("") : '<p class="empty">No useful memories yet.</p>';
}

function radarItems() {
  const target = currentTarget();
  const active = canonicalTarget();
  const result = new Map();
  rankedDetections().forEach(item => {
    const coords = memoryCoordinates(item);
    const distance = numeric(item.distance_m);
    const isTarget = labelKey(item.label) === target;
    if (coords.x !== null && coords.z !== null) result.set(labelKey(item.label), {label:item.label, ...coords, distance:isTarget && active?.distance !== null ? active.distance : distance, visible:true, target:isTarget, hazard:isHazard(item)});
  });
  state.memory.forEach(item => {
    const label = item.object || item.label; const key = labelKey(label); if (result.has(key)) return;
    const coords = memoryCoordinates(item); const distance = numeric(item.distance ?? item.distance_m);
    const isTarget = key === target;
    if (coords.x !== null && coords.z !== null) result.set(key, {label, ...coords, distance:isTarget && active?.distance !== null ? active.distance : distance, visible:Boolean(item.visible), target:isTarget, hazard:false});
  });
  return [...result.values()].sort((a,b) => Number(b.target)-Number(a.target) || Number(b.hazard)-Number(a.hazard) || Number(b.visible)-Number(a.visible) || (a.distance ?? 99)-(b.distance ?? 99)).slice(0,5);
}

function renderRadar() {
  const svg = $("spatialRadar");
  const items = radarItems();
  const extent = Math.max(2, ...items.map(item => Math.max(Math.abs(item.x), Math.abs(item.z), item.distance || 0)));
  const labels = [];
  const pointMarkup = items.map(item => {
    const x = Math.max(22, Math.min(338, 180 + item.x / extent * 145));
    const y = Math.max(18, Math.min(217, 183 - item.z / extent * 150));
    let labelY = y - 10;
    labels.forEach(previous => { if (Math.abs(previous.x - x) < 76 && Math.abs(previous.y - labelY) < 15) labelY += labelY < 185 ? 16 : -16; });
    labels.push({x, y:labelY});
    const colour = item.hazard ? "#ed5f62" : item.target ? "#e3a83d" : item.visible ? "#74c8e8" : "#69767d";
    const fill = item.visible || item.target || item.hazard ? colour : "#0c1012";
    return `<g><circle cx="${x}" cy="${y}" r="${item.target ? 8 : 5}" fill="${fill}" stroke="${colour}" stroke-width="${item.target ? 2 : 1.5}"/><line x1="${x}" y1="${y}" x2="${x}" y2="${labelY + 3}" stroke="${colour}" stroke-opacity=".45"/><text x="${x}" y="${labelY}" text-anchor="middle" fill="#dce3e6" font-size="10" font-family="system-ui">${escapeHtml(titleCase(item.label))}${item.distance === null ? "" : ` ${escapeHtml(distanceText(item.distance))}`}</text></g>`;
  }).join("");
  svg.innerHTML = `<g fill="none" stroke="#263139" stroke-width="1"><path d="M45 183a135 135 0 0 1 270 0"/><path d="M90 183a90 90 0 0 1 180 0"/><path d="M135 183a45 45 0 0 1 90 0"/><line x1="180" y1="18" x2="180" y2="183" stroke-dasharray="3 5"/></g><text x="180" y="13" text-anchor="middle" fill="#69767d" font-size="9" font-family="system-ui">AHEAD</text>${pointMarkup}<g><path d="M180 172l-8 15h16z" fill="#f3f6f7"/><text x="180" y="205" text-anchor="middle" fill="#f3f6f7" font-size="10" font-weight="700" font-family="system-ui">YOU</text></g>`;
}

function renderCameraOverlay() {
  const image = $("camera"), canvas = $("overlay"), rect = image.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1; canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const context = canvas.getContext("2d"); context.scale(dpr, dpr);
  const target = currentTarget(); const mode = uiMode();
  const active = canonicalTarget();
  let items = rankedDetections().filter(item => item.bbox);
  if (mode === "navigate") items = items.filter(item => labelKey(item.label) === target || isHazard(item));
  else if (mode === "find") items = items.filter(item => labelKey(item.label) === target || isHazard(item));
  else items = items.filter(item => isHazard(item) || Number(item.confidence || 0) >= .35).slice(0, 3);
  const sourceWidth = items[0]?.frame_width || image.naturalWidth || 640;
  const sourceHeight = items[0]?.frame_height || image.naturalHeight || 480;
  const scale = Math.max(rect.width/sourceWidth, rect.height/sourceHeight);
  const offsetX = (rect.width-sourceWidth*scale)/2, offsetY = (rect.height-sourceHeight*scale)/2;
  items.slice(0, 3).forEach(item => {
    const box = item.bbox; const x=offsetX+box.x1*scale, y=offsetY+box.y1*scale, width=(box.x2-box.x1)*scale, height=(box.y2-box.y1)*scale;
    const hazard = isHazard(item), targetItem = labelKey(item.label) === target;
    const colour = hazard ? "#ed5f62" : targetItem ? "#e3a83d" : "#74c8e8";
    const corner = Math.max(12, Math.min(24,width*.16,height*.16)); context.strokeStyle=colour; context.lineWidth=hazard||targetItem?2.5:1.5;
    context.beginPath(); [[x,y,1,1],[x+width,y,-1,1],[x+width,y+height,-1,-1],[x,y+height,1,-1]].forEach(([px,py,sx,sy])=>{context.moveTo(px,py+sy*corner);context.lineTo(px,py);context.lineTo(px+sx*corner,py)}); context.stroke();
    const displayDistance = targetItem && active?.distance !== null ? active.distance : item.distance_m;
    const label=`${hazard?"⚠ ":""}${titleCase(item.label)} · ${distanceText(displayDistance)}`; context.font=targetItem?"700 13px system-ui":"600 12px system-ui"; const textWidth=context.measureText(label).width+13;
    context.fillStyle="rgba(5,8,10,.88)"; context.fillRect(x,Math.max(0,y-23),textWidth,21); context.fillStyle=hazard?"#fff0f0":"#f3f6f7"; context.fillText(label,x+6,Math.max(14,y-8));
  });
}

function renderAll() {
  renderMode(); renderGoal(); renderGuidance(); renderNearby(); renderMemory(); renderRadar(); renderCameraOverlay();
}

function setVoiceState(name) {
  const labels = {ready:"Tap the microphone and speak naturally.", listening:"Listening…", processing:"Understanding your request…", speaking:"DrishtiSense is speaking…", unavailable:"Voice recognition is unavailable in this browser. You can type instead."};
  $("voiceState").textContent = labels[name] || name;
  $("voiceButton").classList.toggle("listening", name === "listening");
  $("waveform").classList.toggle("active", ["listening", "speaking"].includes(name));
}

function speak(text, force = false) {
  const clean = cleanAssistantText(text); if (!clean || !("speechSynthesis" in window)) return;
  const now = Date.now(); if (!force && clean === state.lastSpoken && now-state.lastSpokenAt < 4000) return;
  state.lastSpoken=clean; state.lastSpokenAt=now; speechSynthesis.cancel(); const utterance=new SpeechSynthesisUtterance(clean); utterance.rate=.96;
  utterance.onstart=()=>setVoiceState("speaking"); utterance.onend=()=>setVoiceState("ready"); speechSynthesis.speak(utterance);
}

function showResponse(message, shouldSpeak = true) {
  clearTimeout(state.queryTimer); state.queryTimer=null;
  const raw = typeof message === "string" ? message : message?.text;
  let text = cleanAssistantText(raw || "");
  if (!text && message && typeof message === "object") {
    const label = message.target || message.label || "object"; const direction = directionText(message.direction || message.clock_direction || "nearby");
    text = `Your ${label} is ${direction.toLowerCase()}${numeric(message.distance_m) === null ? "." : `, about ${Number(message.distance_m).toFixed(1)} metres away.`}`;
  }
  if (!text) text="I could not get a reliable answer. Please scan the room and try again.";
  const requested = normaliseRequestedObject(message?.object || requestedObjectFromQuery(message?.target));
  if (!state.goal) {
    clearTimeout(state.findTimer);
    state.finding = false;
    if (requested) {
      const navigation = message?.navigation || {};
      const distance = numeric(navigation.distance_m ?? message?.active_target?.distance ?? message?.distance_m);
      const found = distance !== null || Boolean(message?.active_target);
      state.findTarget = requested;
      state.searchOutcome = {
        status: found ? "found" : "not_found", target: message?.object || requested,
        distance, direction: navigation.direction || (navigation.clock_direction ? clockDirection(navigation.clock_direction) : message?.active_target?.direction),
        visible: navigation.visible ?? message?.visible ?? message?.active_target?.visible,
      };
    } else {
      state.findTarget = "";
      state.searchOutcome = null;
    }
  }
  $("assistantResponse").textContent=text; setVoiceState("ready"); if (shouldSpeak) speak(text);
  renderAll();
}

function sendQuery(text) {
  const query=String(text||"").trim(); if (!query) return;
  const requested = requestedObjectFromQuery(query);
  $("userUtterance").textContent=`“${query}”`; $("query").value=""; setVoiceState("processing");

  // The live detection list is already verified by the backend. If the
  // requested object is currently drawn on screen, answer from that same
  // data immediately instead of entering an unnecessary search workflow.
  const visible=requested ? rankedDetections().find(item=>labelKey(item.label)===labelKey(requested)) : null;
  if (visible) {
    const distance=numeric(visible.distance_m);
    const direction=directionText(visible.direction||clockDirection(visible.clock_direction));
    const wornEyeglasses=requested==="eyeglasses"&&visible.source==="opencv-eyeglasses"&&(distance===null||distance<=1.2);
    const response=wornEyeglasses?"You are wearing your eyeglasses; they are on your face.":`Your ${requested} is ${direction.toLowerCase()}${distance===null?".":`, about ${distance.toFixed(1)} metres away.`}`;
    showResponse({
      type:"response",text:response,target:query,object:requested,visible:true,distance_m:distance,
      navigation:{distance_m:distance,direction,visible:true},critic_approved:true,
    });
    // The instant UI answer must still reach the backend. It promotes the
    // verified track into world memory and starts turn-by-turn guidance.
    if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify({type:"query",text:query}));
    return;
  }

  $("assistantResponse").textContent=requested ? "Searching the live camera and memory…" : "Understanding your request…";
  clearTimeout(state.queryTimer);
  // The server bounds focused search time. This timer is only a connection
  // health message; it is not presented as an object-search answer.
  state.queryTimer=setTimeout(()=>{
    state.queryTimer=null;
    $("assistantResponse").textContent=state.socketOpen
      ? `I could not confirm ${titleCase(requested||"that object")} in this scan. Move the camera slowly and try once more.`
      : "The live connection is unavailable. Reconnecting now…";
    setVoiceState("ready");
  },26000);
  if (requested) activateFindMode(requested);
  if (ws.readyState !== WebSocket.OPEN) { showResponse("The live connection is unavailable. Please try again in a moment.", false); return; }
  ws.send(JSON.stringify({type:"query",text:query}));
}

function setupVoice() {
  const Recognition=window.SpeechRecognition||window.webkitSpeechRecognition;
  if (!Recognition) { $("voiceButton").onclick=()=>setVoiceState("unavailable"); return; }
  const recognition=new Recognition(); state.recognition=recognition; recognition.lang=navigator.language||"en-US"; recognition.interimResults=true; recognition.continuous=false;
  recognition.onstart=()=>setVoiceState("listening");
  recognition.onresult=event=>{ let transcript="",finalText=""; for(let i=event.resultIndex;i<event.results.length;i++){transcript+=event.results[i][0].transcript;if(event.results[i].isFinal)finalText+=event.results[i][0].transcript;} $("userUtterance").textContent=`“${transcript.trim()}”`; if(finalText.trim())sendQuery(finalText); };
  recognition.onerror=event=>setVoiceState(event.error==="not-allowed"?"Microphone permission was denied. You can type instead.":"I could not hear that. Please try again."); recognition.onend=()=>{if($("voiceButton").classList.contains("listening"))setVoiceState("ready")};
  $("voiceButton").onclick=()=>{ if($("voiceButton").classList.contains("listening"))recognition.stop(); else recognition.start(); };
}

function renderFocused() {
  const panel=$("focusedResult"), content=$("focusedContent"); clearTimeout(state.focusedTimer);
  if (!state.focused) { panel.hidden=true; return; } panel.hidden=false;
  if (state.focused.loading) { content.innerHTML=`<h3>Searching for ${escapeHtml(titleCase(state.findTarget))}</h3><p>Move the camera slowly across the object.</p>`; return; }
  if (state.focused.error) { content.innerHTML=`<h3>Scan unavailable</h3><p>${escapeHtml(state.focused.error)}</p>`; return; }
  const results=Array.isArray(state.focused)?state.focused.slice().sort((a,b)=>Number(b.confidence||0)-Number(a.confidence||0)):[];
  if (!results.length) content.innerHTML=`<h3>Not found</h3><p>No ${escapeHtml(state.findTarget||"matching object")} is visible yet.</p>`;
  else { const result=results[0]; content.innerHTML=`<h3>${escapeHtml(titleCase(result.label))} found</h3><p>${escapeHtml(distanceText(result.distance_m))} · ${escapeHtml(clockDirection(result.clock_direction))}</p>`; }
  state.focusedTimer=setTimeout(()=>{panel.hidden=true;state.finding=false;if(!state.goal)state.findTarget="";renderAll()},9000);
}

async function runDetection(classes, keepScanning=true) {
  if (!classes.length) return;
  classes=classes.map(normaliseRequestedObject).filter(Boolean);
  if (!classes.length) return;
  const isRoomScan=['home','room'].includes(labelKey(classes[0]));
  state.findTarget=isRoomScan?"":classes[0]; state.finding=!isRoomScan; state.focused=isRoomScan?null:{loading:true}; renderFocused(); renderAll();
  if (ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify({type:"set_open_vocab",classes:keepScanning?classes:[]}));
  try { const response=await fetch("/detect-now",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({classes})}); const body=await response.json(); if(!response.ok)throw new Error(body.error||`Server returned ${response.status}`); state.focused=body.error?{error:body.error}:body.detections||[]; diagnostics("focused_scan",body); renderFocused(); await refreshData(); }
  catch(error){state.focused={error:`Focused scan failed: ${error.message}`};renderFocused()}
}

async function refreshWorldMemory() {
  try { const body=await fetch("/world-memory").then(response=>response.json()); state.memory=body.objects||[]; renderMemory(); renderRadar(); } catch (_) {}
}

async function refreshData() {
  if (state.refreshInFlight) return;
  state.refreshInFlight=true;
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),5000);
  try {
    const [status,memory,path,goal]=await Promise.all(["/status","/world-memory","/safe-path","/goal"].map(url=>fetch(url,{signal:controller.signal}).then(response=>response.json())));
    state.memory=memory.objects||[]; if(!path.error)state.safePath=path; const previousTarget=state.findTarget; const goalSnapshot=normalizeGoal(goal); if(goalSnapshot)state.goal={...(state.goal||{}),...goalSnapshot}; else if(!state.goal||["idle","cancelled"].includes(state.goal.status))state.goal=null;
    if(state.goal?.target && previousTarget!==state.goal.target && ws.readyState===WebSocket.OPEN){state.findTarget=state.goal.target;ws.send(JSON.stringify({type:"set_open_vocab",classes:[state.findTarget]}));}
    $("cameraState").textContent=status.camera?"Live camera":"Camera unavailable"; renderAll();
  } catch (_) {
    if(!state.socketOpen){$("connection").className="connection off";$("connection").querySelector("span").textContent="Reconnecting"}
  } finally { clearTimeout(timeout); state.refreshInFlight=false; }
}

function diagnostics(kind,data){state.lastMessage={kind,data,received_at:new Date().toISOString()};$("diagnostics").textContent=JSON.stringify(state.lastMessage,null,2)}

const camera=$("camera");
camera.onload=()=>{$("cameraState").textContent="Live camera";renderCameraOverlay()};
camera.onerror=()=>{$("cameraState").textContent="Camera frame unavailable"};
const wsScheme=location.protocol==="https:"?"wss":"ws";
let ws=null, reconnectTimer=null, reconnectAttempt=0;

function handleSocketMessage(event) {
  let message;
  try { message=JSON.parse(event.data); }
  catch (_) { diagnostics("socket_error",{message:"The server sent an invalid message."}); return; }
  diagnostics(message.type,message);
  if(message.type==="frame"){camera.src=`data:image/jpeg;base64,${message.jpeg_b64}`}
  else if(message.type==="vision_update"){state.detections=message.detections||[];if("target_state" in message)state.targetState=message.target_state;state.safePath=message.safe_path||state.safePath;renderAll()}
  else if(message.type==="vision_error"){$("cameraState").textContent="Vision temporarily unavailable"}
  else if(message.type==="search_status"){
    if(message.status==="verified") { clearTimeout(state.queryTimer); state.queryTimer=null; }
    state.finding=message.status!=="verified"; state.searchOutcome=null;
    $("assistantResponse").textContent=message.text||"Searching the live camera…";
    setVoiceState(message.status==="verified"?"ready":"processing");
    renderAll();
  }
  else if(message.type==="response"){showResponse(message)}
  else if(message.type==="error"){showResponse(message.message||"The request could not be completed.",false)}
  else if(message.type==="goal_update"){
    const previous=state.lastGoalCommand, previousTarget=state.findTarget; state.lastGoalCommand=message.command||message.status; state.goal=["idle","cancelled"].includes(message.status)?null:message; state.finding=false; state.findTarget=["idle","cancelled"].includes(message.status)?"":(message.target||state.findTarget);
    if (state.goal && previousTarget!==state.findTarget && ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify({type:"set_open_vocab",classes:[state.findTarget]}));
    if(previous!==state.lastGoalCommand||["blocked","complete","lost","unreliable"].includes(message.status))showResponse(message,true); renderAll();
  }
  else if(message.type==="safety_alert"){
    state.safety=message;clearTimeout(state.safetyTimer);state.safetyTimer=setTimeout(()=>{state.safety=null;renderAll()},4500);renderAll();if(message.message)speak(message.message,true);
  }
  else if(message.type==="memory_update"){refreshWorldMemory()}
  else if(message.type==="world_update"&&message.goal){state.goal=normalizeGoal(message.goal)||state.goal;renderAll()}
}

function connectSocket() {
  if (ws && [WebSocket.OPEN,WebSocket.CONNECTING].includes(ws.readyState)) return;
  clearTimeout(reconnectTimer);
  const socket=new WebSocket(`${wsScheme}://${location.host}/ws`);
  ws=socket;
  socket.onopen=()=>{
    if (ws!==socket) return;
    reconnectAttempt=0; state.socketOpen=true;
    $("connection").className="connection live";$("connection").querySelector("span").textContent="Live";
    const target=currentTarget();
    if (target && state.finding) socket.send(JSON.stringify({type:"set_open_vocab",classes:[target]}));
    refreshData();
  };
  socket.onmessage=handleSocketMessage;
  socket.onerror=()=>{if(ws===socket){state.socketOpen=false;socket.close()}};
  socket.onclose=()=>{
    if (ws!==socket) return;
    state.socketOpen=false;$("connection").className="connection off";$("connection").querySelector("span").textContent="Reconnecting";
    const delay=Math.min(5000,500*(2**reconnectAttempt)); reconnectAttempt+=1;
    reconnectTimer=setTimeout(connectSocket,delay);
  };
}

$("queryForm").onsubmit=event=>{event.preventDefault();sendQuery($("query").value)};
$("targetButton").onclick=()=>{const classes=$("targets").value.split(",").map(value=>value.trim()).filter(Boolean);const small=new Set(["key","keys","keychain","cell phone","remote","wallet","toothbrush","pen"]);runDetection(classes,!classes.some(item=>small.has(item.toLowerCase())))};
$("homeScanButton").onclick=()=>runDetection(["home"],false);
$("viewAll").onclick=()=>{state.showAll=!state.showAll;renderNearby()};
$("dismissFocused").onclick=()=>{clearTimeout(state.focusedTimer);clearTimeout(state.findTimer);$("focusedResult").hidden=true;state.finding=false;if(!state.goal)state.findTarget="";if(ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify({type:"set_open_vocab",classes:[]}));renderAll()};
$("cancelGoal").onclick=async()=>{try{await fetch("/goal",{method:"DELETE"})}catch(_){}clearTimeout(state.findTimer);state.goal=null;state.findTarget="";if(ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify({type:"set_open_vocab",classes:[]}));renderAll()};

connectSocket(); setupVoice(); renderAll(); refreshData();
setInterval(()=>{if(!state.socketOpen)camera.src=`/camera.jpg?t=${Date.now()}`},600);
setInterval(refreshData,3000);
setInterval(()=>{if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify({type:"ping"}))},15000);
window.addEventListener("resize",renderCameraOverlay);
