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

// ── In-memory session storage (client side) ────────────────────
let baselineBiomarkers = null;

// ── Math Helpers ───────────────────────────────────────────────
function mean(arr) {
    if (!arr || arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr, meanVal) {
    if (!arr || arr.length === 0) return 0;
    const avg = meanVal !== undefined ? meanVal : mean(arr);
    const sqDiff = arr.map(x => (x - avg) * (x - avg));
    return Math.sqrt(mean(sqDiff));
}

function computeBiomarkers(velocities, accels, blinkRateHz) {
    const v = velocities;
    const a = accels;
    const meanVel = mean(v);
    const stdVel = std(v, meanVel);
    const velCv = stdVel / (meanVel + 1e-6);
    
    // np.diff(v)
    const velDiff = [];
    for (let i = 0; i < v.length - 1; i++) {
        velDiff.push(v[i+1] - v[i]);
    }
    
    // np.sign(vel_diff)
    const signs = velDiff.map(x => {
        if (x > 0) return 1;
        if (x < 0) return -1;
        return 0;
    });
    
    // np.diff(np.sign(vel_diff)) != 0
    let signChanges = 0;
    for (let i = 0; i < signs.length - 1; i++) {
        if (signs[i+1] !== signs[i]) {
            signChanges++;
        }
    }
    const tremorIndex = signChanges / Math.max(velDiff.length - 1, 1);
    
    // fixation_mask = v < (mean_vel * 0.3)
    const fixationVelocities = v.filter(x => x < meanVel * 0.3);
    const fixationInstability = fixationVelocities.length > 3 ? std(fixationVelocities) : 0.0;
    
    const absAcc = a.map(x => Math.abs(x));
    const meanAccAbs = mean(absAcc);
    const accelIrregularity = meanAccAbs / (meanVel + 1e-6);
    
    // find_peaks(v, height=mean_vel * 1.5, distance=3)
    const peakHeight = meanVel * 1.5;
    const peaks = [];
    for (let i = 1; i < v.length - 1; i++) {
        if (v[i] > peakHeight && v[i] > v[i-1] && v[i] > v[i+1]) {
            if (peaks.length === 0 || (i - peaks[peaks.length - 1]) >= 3) {
                peaks.push(i);
            }
        }
    }
    
    return {
        mean_vel: meanVel,
        std_vel: stdVel,
        vel_cv: velCv,
        tremor_index: tremorIndex,
        fixation_instability: fixationInstability,
        accel_irregularity: accelIrregularity,
        max_vel: v.length > 0 ? Math.max(...v) : 0,
        saccade_count: peaks.length,
        blink_rpm: blinkRateHz * 60.0
    };
}

function classifyAbsolute(bm) {
    const v_cv = bm.vel_cv;
    const ti   = bm.tremor_index;
    const ai   = bm.accel_irregularity;
    const bpm  = bm.blink_rpm;

    const cv_score     = Math.min((Math.max(v_cv - 1.2, 0) / 2.5) * 35, 35);
    const tremor_score = Math.min((Math.max(ti - 0.55, 0) / 0.30) * 35, 35);
    const acc_score    = Math.min((Math.max(ai - 3.5, 0) / 7.0)  * 20, 20);
    const score = cv_score + tremor_score + acc_score;

    if (bpm > 35 && score < 52) {
        return {
            prediction: 'Minor Issues',
            probability: Math.min(0.45 + (bpm - 30) / 100, 0.72),
            score: score,
            reason: `High blink rate (${bpm.toFixed(0)} bpm) — possible eye irritation.`
        };
    }

    if (score >= 55) {
        return {
            prediction: 'Micro-Trauma Detected',
            probability: Math.min(0.62 + score / 300, 0.96),
            score: score,
            reason: `Erratic saccadic profile (CV=${v_cv.toFixed(2)}, tremor=${ti.toFixed(2)}).`
        };
    }
    if (score >= 28) {
        return {
            prediction: 'Minor Issues',
            probability: Math.min(0.35 + score / 150, 0.72),
            score: score,
            reason: `Mildly irregular movement (tremor=${ti.toFixed(2)}) — possible fatigue.`
        };
    }
    return {
        prediction: 'Healthy Eyes',
        probability: Math.max(0.08, 0.18 - score / 180),
        score: score,
        reason: `Normal saccadic profile (score=${score.toFixed(1)}/55).`
    };
}

function classifyWithBaseline(bm, baseline) {
    function dev(cur, base, natural_spread) {
        return (cur - base) / Math.max(natural_spread, 1e-6);
    }

    const tremor_dev = dev(bm.tremor_index,
                           baseline.tremor_index,
                           Math.max(baseline.tremor_index * 0.25, 0.04));

    const cv_dev     = dev(bm.vel_cv,
                           baseline.vel_cv,
                           Math.max(baseline.vel_cv * 0.35, 0.15));

    const acc_dev    = dev(bm.accel_irregularity,
                           baseline.accel_irregularity,
                           Math.max(baseline.accel_irregularity * 0.30, 0.5));

    const bpm        = bm.blink_rpm;
    const base_bpm   = Math.max(baseline.blink_rpm, 10.0);
    const blink_dev  = (bpm - base_bpm) / Math.max(base_bpm * 0.5, 5.0);

    // Scoring
    let score = 0.0;
    const reasons = [];

    if (tremor_dev > 3.5) {
        score += 45; reasons.push(`severe tremor increase (${bm.tremor_index.toFixed(2)} vs baseline ${baseline.tremor_index.toFixed(2)})`);
    } else if (tremor_dev > 2.0) {
        score += 25; reasons.push(`elevated tremor (${bm.tremor_index.toFixed(2)} vs ${baseline.tremor_index.toFixed(2)})`);
    } else if (tremor_dev > 1.0) {
        score += 10;
    }

    if (cv_dev > 3.5) {
        score += 35; reasons.push(`very erratic saccades (CV=${bm.vel_cv.toFixed(2)})`);
    } else if (cv_dev > 2.0) {
        score += 18; reasons.push(`irregular saccadic pattern`);
    } else if (cv_dev > 1.0) {
        score += 8;
    }

    if (acc_dev > 3.5) {
        score += 20; reasons.push('high jerk irregularity');
    } else if (acc_dev > 2.0) {
        score += 10;
    }

    if (blink_dev > 2.0 && bpm > 28) {
        score += 12; reasons.push(`elevated blink rate (${bpm.toFixed(0)} bpm vs baseline ${base_bpm.toFixed(0)})`);
    }

    if (score >= 50) {
        const reason = reasons.length > 0 ? 'Significant deviation from your baseline: ' + reasons.join(', ')
                                          : 'Multiple abnormal biomarkers vs your personal baseline.';
        return {
            prediction: 'Micro-Trauma Detected',
            probability: Math.min(0.65 + score / 300, 0.96),
            score: score,
            reason: reason
        };
    } else if (score >= 18) {
        const reason = reasons.length > 0 ? 'Mild deviation from your baseline: ' + reasons.join(', ')
                                          : 'Slightly elevated biomarkers vs your personal baseline.';
        return {
            prediction: 'Minor Issues',
            probability: Math.min(0.35 + score / 150, 0.72),
            score: score,
            reason: reason
        };
    } else {
        return {
            prediction: 'Healthy Eyes',
            probability: Math.max(0.06, 0.18 - score / 200),
            score: score,
            reason: `Eye movement consistent with your personal baseline (deviation=${score.toFixed(0)}).`
        };
    }
}

// ── Local API Simulation ───────────────────────────────────────
async function sendCalibrate(windows) {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const all_bm = [];
    for (const window of windows) {
        if (window.length === 50) {
            const velocities = window.map(w => w[0]);
            const accels = window.map(w => w[1]);
            const bm = computeBiomarkers(velocities, accels, getBlinkRate());
            all_bm.push(bm);
        }
    }
    
    if (all_bm.length === 0) {
        return { error: 'No valid windows' };
    }
    
    const keys = Object.keys(all_bm[0]);
    const baseline = {};
    for (const k of keys) {
        const values = all_bm.map(b => b[k]);
        baseline[k] = mean(values);
    }
    
    baselineBiomarkers = baseline;
    return { status: 'ok', baseline: baseline };
}

async function sendPredict(sequences) {
    await new Promise(resolve => setTimeout(resolve, 50));
    
    if (sequences.length !== 50) {
        return { error: `Invalid shape ${sequences.length}` };
    }
    
    const velocities = sequences.map(s => s[0]);
    const accels = sequences.map(s => s[1]);
    
    const vMean = mean(velocities);
    const vStd = std(velocities, vMean);
    if (vMean < 0.3 && vStd < 0.3) {
        return {
            prediction: 'Healthy Eyes',
            probability: 0.08,
            score: 0,
            reason: 'Very stable fixation detected.',
            calibrated: baselineBiomarkers !== null
        };
    }
    
    const bm = computeBiomarkers(velocities, accels, getBlinkRate());
    let result;
    if (baselineBiomarkers) {
        result = classifyWithBaseline(bm, baselineBiomarkers);
        result.calibrated = true;
    } else {
        result = classifyAbsolute(bm);
        result.calibrated = false;
    }
    
    return result;
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
