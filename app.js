// ── State machine ──────────────────────────────────────────────
// IDLE → CALIBRATING → ANALYZING → STOPPED
let appState = 'IDLE';
let sessionId = null;
const CALIBRATION_WINDOWS = 4;   // ~6 seconds at 30 fps
let calibrationWindowsCollected = [];

// DOM refs
const videoElement   = document.getElementById('input_video');
const canvasElement  = document.getElementById('output_canvas');
const canvasCtx      = canvasElement.getContext('2d');
const startBtn       = document.getElementById('startBtn');
const stopBtn        = document.getElementById('stopBtn');
const recalBtn       = document.getElementById('recalBtn');
const statusText     = document.getElementById('statusText');
const statusSubText  = document.getElementById('statusSubText');
const probValue      = document.getElementById('probValue');
const framesValue    = document.getElementById('framesValue');
const blinkRateValue = document.getElementById('blinkRateValue');
const calibBar       = document.getElementById('calibBar');
const calibSection   = document.getElementById('calibSection');
const ring           = document.querySelector('.ring');

// ── Tracking state ─────────────────────────────────────────────
let sequenceBuffer = [];
let lastPoint = null, lastTime = null, lastVelocity = null;
let blinkCount = 0, sessionStartTime = null;
let eyeClosedFrames = 0, eyeWasOpen = true, isEyeClosed = false;
const EAR_THRESHOLD   = 0.21;
const CLOSED_LIMIT    = 5;

// ── Velocity chart ─────────────────────────────────────────────
const ctx = document.getElementById('velocityChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(60).fill(''),
        datasets: [{
            data: Array(60).fill(0),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,0.08)',
            borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { y: { display: false }, x: { display: false } },
        plugins: { legend: { display: false } }
    }
});

function pushChart(v) {
    const d = chart.data.datasets[0].data;
    d.push(Math.min(v, 60));
    if (d.length > 60) d.shift();
    chart.update('none');
}

// ── Eye helpers ────────────────────────────────────────────────
const L_EAR = [362, 385, 387, 263, 373, 380];
const R_EAR = [33,  160, 158, 133, 153, 144];
const L_CTR = [362, 263, 385, 380];
const R_CTR = [33,  133, 160, 144];

function euc(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function ear(lm, idx) {
    return (euc(lm[idx[1]], lm[idx[5]]) + euc(lm[idx[2]], lm[idx[4]])) /
           (2 * euc(lm[idx[0]], lm[idx[3]]));
}

function center(lm, idx) {
    let x = 0, y = 0;
    idx.forEach(i => { x += lm[i].x; y += lm[i].y; });
    return { x: x / idx.length, y: y / idx.length };
}

// ── Helpers ────────────────────────────────────────────────────
function generateId() { return Math.random().toString(36).substr(2, 9); }

function getBlinkRate() {
    const sec = sessionStartTime ? (Date.now() - sessionStartTime) / 1000 : 1;
    return blinkCount / Math.max(sec, 1);
}

// ── API calls ──────────────────────────────────────────────────
async function sendCalibrate(windows) {
    const res = await fetch('/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, windows, blink_rate: getBlinkRate() })
    });
    return res.json();
}

async function sendPredict(sequences) {
    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, sequences, blink_rate: getBlinkRate() })
    });
    return res.json();
}

// ── UI updaters ────────────────────────────────────────────────
function setRing(cls, text, sub, color) {
    ring.className = 'ring' + (cls ? ' ' + cls : '');
    statusText.innerText = text;
    statusText.style.color = color || 'var(--text-light)';
    statusSubText.innerText = sub;
}

function updateCalibProgress() {
    const pct = Math.round((calibrationWindowsCollected.length / CALIBRATION_WINDOWS) * 100);
    calibBar.style.width = pct + '%';
    framesValue.innerText = `Cal ${pct}%`;
}

// ── Face mesh callback ─────────────────────────────────────────
function onResults(results) {
    if (appState === 'IDLE' || appState === 'STOPPED') return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (!results.multiFaceLandmarks?.length) {
        setRing('', 'No Face Detected', 'Move closer to the camera', 'var(--text-muted)');
        canvasCtx.restore(); return;
    }

    const lm = results.multiFaceLandmarks[0];
    drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE,  { color: '#3b82f6', lineWidth: 1 });
    drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#3b82f6', lineWidth: 1 });

    const avgEAR = (ear(lm, L_EAR) + ear(lm, R_EAR)) / 2;

    if (avgEAR < EAR_THRESHOLD) {
        eyeClosedFrames++;
        isEyeClosed = true;
        if (eyeClosedFrames === 2 && eyeWasOpen) { blinkCount++; eyeWasOpen = false; }
        if (eyeClosedFrames > CLOSED_LIMIT) {
            setRing('closed', 'Eyes Closed', 'Open eyes to resume tracking', 'var(--text-muted)');
            canvasCtx.restore(); return;
        }
    } else {
        eyeClosedFrames = 0; isEyeClosed = false; eyeWasOpen = true;
    }

    const bpm = (getBlinkRate() * 60).toFixed(1);
    blinkRateValue.innerText = `${bpm} bpm`;

    // Compute velocity
    const cur = { x: (center(lm, L_CTR).x + center(lm, R_CTR).x) / 2,
                  y: (center(lm, L_CTR).y + center(lm, R_CTR).y) / 2 };
    const now = performance.now();

    if (lastPoint && lastTime && !isEyeClosed) {
        const dt = (now - lastTime) / 1000;
        if (dt > 0) {
            const dx = (cur.x - lastPoint.x) * 100;
            const dy = (cur.y - lastPoint.y) * 100;
            const vel  = Math.hypot(dx, dy) / dt;
            const acc  = lastVelocity !== null ? (vel - lastVelocity) / dt : 0;

            pushChart(vel);
            sequenceBuffer.push([vel, acc]);
            lastVelocity = vel;

            if (appState === 'CALIBRATING') {
                const needed = 50;
                framesValue.innerText = `${Math.min(sequenceBuffer.length, needed)} / ${needed}`;
                if (sequenceBuffer.length >= needed) {
                    calibrationWindowsCollected.push([...sequenceBuffer]);
                    sequenceBuffer = [];
                    updateCalibProgress();

                    if (calibrationWindowsCollected.length >= CALIBRATION_WINDOWS) {
                        calibSection.style.display = 'none';
                        setRing('', 'Finalising baseline…', 'Please wait', 'var(--text-muted)');
                        sendCalibrate(calibrationWindowsCollected).then(() => {
                            appState = 'ANALYZING';
                            setRing('baseline', '✓ Baseline Ready', 'Now monitoring for deviations from your normal', 'var(--success)');
                            framesValue.innerText = '0 / 50';
                            recalBtn.disabled = false;
                        });
                    } else {
                        const pct = Math.round((calibrationWindowsCollected.length / CALIBRATION_WINDOWS) * 100);
                        setRing('', `Calibrating… ${pct}%`, 'Keep looking naturally at the screen', 'var(--text-muted)');
                    }
                }
            } else if (appState === 'ANALYZING') {
                framesValue.innerText = `${Math.min(sequenceBuffer.length, 50)} / 50`;
                if (sequenceBuffer.length >= 50) {
                    const win = [...sequenceBuffer];
                    sequenceBuffer = sequenceBuffer.slice(25);   // rolling window
                    sendPredict(win).then(data => {
                        if (data.prediction) updateUI(data);
                    });
                }
            }
        }
    }

    lastPoint = cur;
    lastTime  = now;
    canvasCtx.restore();
}

function updateUI(data) {
    const pct = data.probability !== undefined ? (data.probability * 100).toFixed(1) + '%' : '—';
    probValue.innerText = pct;
    statusSubText.innerText = data.reason || '';
    const cal = data.calibrated ? '' : ' (no baseline)';

    if (data.prediction === 'Micro-Trauma Detected') {
        setRing('strained', '⚠ Micro-Trauma Detected' + cal, data.reason, 'var(--danger)');
    } else if (data.prediction === 'Minor Issues') {
        setRing('warning', '~ Minor Eye Issues' + cal, data.reason, 'var(--warning)');
    } else {
        setRing('baseline', '✓ Healthy Eyes' + cal, data.reason, 'var(--success)');
    }
}

// ── MediaPipe ─────────────────────────────────────────────────
const faceMesh = new FaceMesh({ locateFile: f =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true,
    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
faceMesh.onResults(onResults);

let camera;

function resetTracking() {
    sequenceBuffer = []; lastPoint = null; lastTime = null; lastVelocity = null;
    blinkCount = 0; eyeClosedFrames = 0; eyeWasOpen = true; isEyeClosed = false;
}

// ── Buttons ────────────────────────────────────────────────────
startBtn.addEventListener('click', () => {
    sessionId = generateId();
    calibrationWindowsCollected = [];
    resetTracking();
    sessionStartTime = Date.now();
    appState = 'CALIBRATING';

    startBtn.disabled = true;
    stopBtn.disabled  = false;
    recalBtn.disabled = true;

    calibSection.style.display = 'block';
    calibBar.style.width = '0%';

    canvasElement.width  = videoElement.clientWidth  || 640;
    canvasElement.height = videoElement.clientHeight || 480;

    setRing('', 'Calibrating… 0%', 'Look naturally at the screen for ~6 seconds', 'var(--text-muted)');

    camera = new Camera(videoElement, {
        onFrame: async () => { if (appState !== 'STOPPED') await faceMesh.send({ image: videoElement }); },
        width: 640, height: 480
    });
    camera.start();
});

stopBtn.addEventListener('click', () => {
    appState = 'STOPPED';
    if (camera) camera.stop();
    resetTracking();
    sessionId = null;
    calibrationWindowsCollected = [];

    startBtn.disabled = false;
    stopBtn.disabled  = true;
    recalBtn.disabled = true;
    calibSection.style.display = 'none';

    setRing('', 'Session Stopped', 'Press Start Tracking to begin a new session', 'var(--text-light)');
    probValue.innerText      = '—';
    framesValue.innerText    = '0 / 50';
    blinkRateValue.innerText = '0 bpm';
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
});

recalBtn.addEventListener('click', () => {
    calibrationWindowsCollected = [];
    resetTracking();
    appState = 'CALIBRATING';
    recalBtn.disabled = true;
    calibSection.style.display = 'block';
    calibBar.style.width = '0%';
    setRing('', 'Re-calibrating… 0%', 'Look naturally at the screen', 'var(--text-muted)');
});
