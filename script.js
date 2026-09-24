// ──────────────────────────────────────────────
//  Sign Language Communication Bridge – script.js
// ──────────────────────────────────────────────

const API_BASE = "http://localhost:5000"; // Flask backend

// ── DOM refs (filled in once the DOM is ready) ─
let video, canvas, ctx, startBtn, stopBtn, clearBtn, speakBtn;
let predictionEl, confidenceEl, sentenceEl, statusDot, statusText;
let historyList, modeToggle, thresholdInput;

// ── State ─────────────────────────────────────
let stream          = null;
let capturing       = false;
let frameTimer      = null;
let sentence        = [];
let lastSign        = null;
let holdFrames      = 0;
const HOLD_NEEDED   = 8;   // frames a sign must persist before it's committed
const FPS           = 10;  // how often we sample (frames / second)
let confidenceThreshold = 0.75;

// ── Bootstrap ─────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Grab every element only after the DOM actually exists
  video          = document.getElementById("webcam");
  canvas         = document.getElementById("overlay");
  startBtn       = document.getElementById("startBtn");
  stopBtn        = document.getElementById("stopBtn");
  clearBtn       = document.getElementById("clearBtn");
  speakBtn       = document.getElementById("speakBtn");
  predictionEl   = document.getElementById("prediction");
  confidenceEl   = document.getElementById("confidence");
  sentenceEl     = document.getElementById("sentence");
  statusDot      = document.getElementById("statusDot");
  statusText     = document.getElementById("statusText");
  historyList    = document.getElementById("historyList");
  modeToggle     = document.getElementById("modeToggle");     // checkbox: word vs letter
  thresholdInput = document.getElementById("threshold");      // confidence slider

  if (!canvas) {
    console.error("[SL Bridge] #overlay canvas not found in the DOM — check the id in your HTML.");
    return;
  }
  ctx = canvas.getContext("2d");

  setStatus("idle");
  bindEvents();
  pingBackend();
});

// ── Event bindings ────────────────────────────
function bindEvents() {
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click",  stopCamera);
  clearBtn.addEventListener("click", clearSentence);
  speakBtn.addEventListener("click", speakSentence);

  if (thresholdInput) {
    thresholdInput.addEventListener("input", () => {
      confidenceThreshold = parseFloat(thresholdInput.value);
      const label = document.getElementById("thresholdLabel");
      if (label) label.textContent = `${Math.round(confidenceThreshold * 100)}%`;
    });
  }
}

// ── Backend health check ───────────────────────
async function pingBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) setStatus("ready");
  } catch {
    setStatus("offline");
  }
}

// ── Camera ────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    capturing = true;
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    setStatus("capturing");

    frameTimer = setInterval(captureFrame, 1000 / FPS);
  } catch (err) {
    showError(`Camera error: ${err.message}`);
  }
}

function stopCamera() {
  capturing = false;
  clearInterval(frameTimer);
  frameTimer = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  startBtn.disabled = false;
  stopBtn.disabled  = true;
  setStatus("ready");
  predictionEl.textContent  = "–";
  confidenceEl.textContent  = "";
}

// ── Frame capture & prediction ────────────────
async function captureFrame() {
  if (!capturing || video.readyState < 2) return;

  // Draw current video frame to an off-screen canvas, get JPEG blob
  const offscreen = document.createElement("canvas");
  offscreen.width  = canvas.width;
  offscreen.height = canvas.height;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(video, 0, 0);
}