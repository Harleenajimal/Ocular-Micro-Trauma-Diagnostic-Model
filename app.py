from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

# In-memory session storage: session_id -> baseline biomarkers
sessions = {}

def compute_biomarkers(velocities, accels, blink_rate_hz):
    from scipy.signal import find_peaks
    v = np.array(velocities, dtype=float)
    a = np.array(accels, dtype=float)
    mean_vel = float(np.mean(v))
    std_vel  = float(np.std(v))
    vel_cv   = std_vel / (mean_vel + 1e-6)
    vel_diff = np.diff(v)
    sign_changes = int(np.sum(np.diff(np.sign(vel_diff)) != 0))
    tremor_index = sign_changes / max(len(vel_diff) - 1, 1)
    fixation_mask = v < (mean_vel * 0.3)
    fixation_instability = float(np.std(v[fixation_mask])) if fixation_mask.sum() > 3 else 0.0
    mean_acc_abs = float(np.mean(np.abs(a)))
    accel_irregularity = mean_acc_abs / (mean_vel + 1e-6)
    peaks, _ = find_peaks(v, height=mean_vel * 1.5, distance=3)
    return {
        'mean_vel': mean_vel,
        'std_vel': std_vel,
        'vel_cv': vel_cv,
        'tremor_index': tremor_index,
        'fixation_instability': fixation_instability,
        'accel_irregularity': accel_irregularity,
        'max_vel': float(np.max(v)),
        'saccade_count': len(peaks),
        'blink_rpm': blink_rate_hz * 60.0
    }


def classify_absolute(bm):
    """Fallback classifier when no baseline is available. Conservative thresholds."""
    score = 0.0
    v_cv = bm['vel_cv']
    ti   = bm['tremor_index']
    ai   = bm['accel_irregularity']
    bpm  = bm['blink_rpm']

    cv_score     = min((max(v_cv - 1.2, 0) / 2.5) * 35, 35)
    tremor_score = min((max(ti - 0.55, 0) / 0.30) * 35, 35)
    acc_score    = min((max(ai - 3.5, 0) / 7.0)  * 20, 20)
    score = cv_score + tremor_score + acc_score

    if bpm > 35 and score < 52:
        return 'Minor Issues', min(0.45 + (bpm - 30) / 100, 0.72), score, \
               f'High blink rate ({bpm:.0f} bpm) — possible eye irritation.'

    if score >= 55:
        return 'Micro-Trauma Detected', min(0.62 + score / 300, 0.96), score, \
               f'Erratic saccadic profile (CV={v_cv:.2f}, tremor={ti:.2f}).'
    if score >= 28:
        return 'Minor Issues', min(0.35 + score / 150, 0.72), score, \
               f'Mildly irregular movement (tremor={ti:.2f}) — possible fatigue.'
    return 'Healthy Eyes', max(0.08, 0.18 - score / 180), score, \
           f'Normal saccadic profile (score={score:.1f}/55).'


def classify_with_baseline(bm, baseline):
    """
    Deviation-based classifier.  Every metric is measured as how many
    'natural-variation units' it has drifted from the user's own baseline.
    """
    def dev(cur, base, natural_spread):
        return (cur - base) / max(natural_spread, 1e-6)

    tremor_dev = dev(bm['tremor_index'],
                     baseline['tremor_index'],
                     max(baseline['tremor_index'] * 0.25, 0.04))

    cv_dev     = dev(bm['vel_cv'],
                     baseline['vel_cv'],
                     max(baseline['vel_cv'] * 0.35, 0.15))

    acc_dev    = dev(bm['accel_irregularity'],
                     baseline['accel_irregularity'],
                     max(baseline['accel_irregularity'] * 0.30, 0.5))

    bpm        = bm['blink_rpm']
    base_bpm   = max(baseline['blink_rpm'], 10.0)
    blink_dev  = (bpm - base_bpm) / max(base_bpm * 0.5, 5.0)

    # Scoring
    score = 0.0
    reasons = []

    if tremor_dev > 3.5:
        score += 45; reasons.append(f'severe tremor increase ({bm["tremor_index"]:.2f} vs baseline {baseline["tremor_index"]:.2f})')
    elif tremor_dev > 2.0:
        score += 25; reasons.append(f'elevated tremor ({bm["tremor_index"]:.2f} vs {baseline["tremor_index"]:.2f})')
    elif tremor_dev > 1.0:
        score += 10

    if cv_dev > 3.5:
        score += 35; reasons.append(f'very erratic saccades (CV={bm["vel_cv"]:.2f})')
    elif cv_dev > 2.0:
        score += 18; reasons.append(f'irregular saccadic pattern')
    elif cv_dev > 1.0:
        score += 8

    if acc_dev > 3.5:
        score += 20; reasons.append('high jerk irregularity')
    elif acc_dev > 2.0:
        score += 10

    if blink_dev > 2.0 and bpm > 28:
        score += 12; reasons.append(f'elevated blink rate ({bpm:.0f} bpm vs baseline {base_bpm:.0f})')

    if score >= 50:
        reason = 'Significant deviation from your baseline: ' + ', '.join(reasons) if reasons \
                 else 'Multiple abnormal biomarkers vs your personal baseline.'
        return 'Micro-Trauma Detected', min(0.65 + score / 300, 0.96), score, reason
    elif score >= 18:
        reason = 'Mild deviation from your baseline: ' + ', '.join(reasons) if reasons \
                 else 'Slightly elevated biomarkers vs your personal baseline.'
        return 'Minor Issues', min(0.35 + score / 150, 0.72), score, reason
    else:
        return 'Healthy Eyes', max(0.06, 0.18 - score / 200), score, \
               f'Eye movement consistent with your personal baseline (deviation={score:.0f}).'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    session_id = data.get('session_id')
    windows    = data.get('windows', [])
    blink_rate = float(data.get('blink_rate', 0))

    if not session_id or not windows:
        return jsonify({'error': 'Missing session_id or windows'}), 400

    all_bm = []
    for window in windows:
        arr = np.array(window)
        if arr.shape == (50, 2):
            bm = compute_biomarkers(arr[:, 0].tolist(), arr[:, 1].tolist(), blink_rate)
            all_bm.append(bm)

    if not all_bm:
        return jsonify({'error': 'No valid windows'}), 400

    # Average biomarkers across calibration windows
    keys     = all_bm[0].keys()
    baseline = {k: float(np.mean([b[k] for b in all_bm])) for k in keys}
    sessions[session_id] = baseline
    return jsonify({'status': 'ok', 'baseline': baseline})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'sequences' not in data:
        return jsonify({'error': 'No sequence data provided'}), 400

    sequences  = np.array(data['sequences'])
    blink_rate = float(data.get('blink_rate', 0))
    session_id = data.get('session_id')

    if sequences.shape != (50, 2):
        return jsonify({'error': f'Invalid shape {sequences.shape}'}), 400

    if sequences[:, 0].mean() < 0.3 and sequences[:, 0].std() < 0.3:
        return jsonify({'prediction': 'Healthy Eyes', 'probability': 0.08,
                        'score': 0, 'reason': 'Very stable fixation detected.'})

    bm       = compute_biomarkers(sequences[:, 0].tolist(), sequences[:, 1].tolist(), blink_rate)
    baseline = sessions.get(session_id) if session_id else None

    if baseline:
        pred, prob, score, reason = classify_with_baseline(bm, baseline)
    else:
        pred, prob, score, reason = classify_absolute(bm)

    return jsonify({'prediction': pred, 'probability': prob,
                    'score': round(score, 1), 'reason': reason,
                    'calibrated': baseline is not None})


@app.route('/test')
def self_test():
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 50)
    results = {}

    # Healthy: smooth sinusoidal
    hv = np.abs(12 * np.sin(0.5 * t)) + np.random.normal(0, 0.4, 50)
    ha = np.diff(np.clip(hv, 0.1, None), prepend=hv[0]) / 0.033
    bm_h = compute_biomarkers(hv.tolist(), ha.tolist(), 0.22)
    # Baseline = same as healthy
    base = bm_h.copy()
    p, prob, sc, r = classify_with_baseline(bm_h, base)
    results['Healthy'] = {'predicted': p, 'correct': p == 'Healthy Eyes', 'score': round(sc, 2), 'reason': r}

    # Minor: moderate irregularity + 33 bpm blink
    mv = np.abs(8 * np.sin(t) + np.random.normal(0, 2.5, 50))
    ma = np.diff(np.clip(mv, 0.1, None), prepend=mv[0]) / 0.033
    bm_m = compute_biomarkers(mv.tolist(), ma.tolist(), 0.55)
    p, prob, sc, r = classify_with_baseline(bm_m, base)
    results['Minor Issues'] = {'predicted': p, 'correct': p == 'Minor Issues', 'score': round(sc, 2), 'reason': r}

    # Trauma: nystagmus-like rapid oscillation
    tv = np.abs(22 * np.sin(8 * t) + 12 * np.sin(13 * t + 1.1)) + np.random.normal(0, 6, 50) + 1.5
    ta = np.diff(tv, prepend=tv[0]) / 0.033
    bm_t = compute_biomarkers(tv.tolist(), ta.tolist(), 0.22)
    p, prob, sc, r = classify_with_baseline(bm_t, base)
    results['Micro-Trauma'] = {'predicted': p, 'correct': p == 'Micro-Trauma Detected', 'score': round(sc, 2), 'reason': r}

    return jsonify({'all_correct': all(v['correct'] for v in results.values()), 'results': results})


if __name__ == '__main__':
    print("Ocular Micro-Trauma Detector starting...")
    print("Self-test: http://127.0.0.1:5000/test")
    app.run(debug=True, port=5000, use_reloader=False)
