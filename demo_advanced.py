"""
ADVANCED DEMO: confidence-gated FSM + temporal multi-frame VLM + failure-type
prompt routing + sim-to-real noise ablation + semantic MCS fault logging.

Built on the same PyBullet + Franka Panda + eye-in-hand pipeline. The VLM has two
interchangeable backends (MockVLM / RealVLM); flip USE_REAL_VLM to run against
Llama 3.2 Vision. All five features work with either backend.

Outputs:
  - console trace of the confidence-gated, routed, temporal recovery loop
  - outputs/sim2real_ablation.png   (noise ablation study)
  - outputs/mcs_faults.jsonl        (semantic fault log)
"""
import os, json, time, math, random
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data

OUT = Path(__file__).resolve().parent / "outputs"
FRAMES_DIR = OUT / "frames"
OUT.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)


# ---------------- config ----------------
USE_REAL_VLM = False
VLM_MODEL = "llama3.2-vision"

PANDA_EE_LINK = 11
PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]
FINGER_OPEN = 0.04
DOWN_ORN = None

CAPTURE_RADIUS = 0.028          # m
DIAGNOSE_HOVER_Z = 0.22
BASE_XY = (0.5, 0.0)
MAX_RECOVERIES = 3

# (1) confidence gating thresholds
CONF_ACT = 0.60                 # >= act on the correction
CONF_REQUERY = 0.35             # in [REQUERY, ACT): re-query with more frames first
                                # < REQUERY: abort (unsafe to act)

# (2) temporal multi-frame
TEMPORAL_FRAMES = 3
CAM_JITTER_M = 0.004            # small pose jitter between frames

# (3) failure taxonomy + prompt routing
F_POSITIONAL = "POSITIONAL_OFFSET"
F_NO_OBJECT = "NO_OBJECT"
F_OCCLUSION = "OCCLUSION"
F_ORIENTATION = "ORIENTATION"
OUT_OF_FRAME_CM = 12.0          # object beyond this at hover -> not in view

BASE_INSTR = ("You are a top-down pick-and-place failure-diagnosis module with a "
              "downward eye-in-hand camera. Image centre = gripper target. "
              "Camera axes: +x image-right, +y image-up. Reason briefly, then output "
              'ONLY JSON: {"reasoning":"..","dx_cm":<n>,"dy_cm":<n>,"confidence":<0-1>}')

# routed, failure-type-specific system prompts (the real model uses these verbatim)
PROMPTS = {
    F_POSITIONAL: BASE_INSTR + " The object is visible but off-centre; give the cm "
                  "offset to ADD to the target to centre on it.",
    F_NO_OBJECT:  BASE_INSTR + " The object may be absent/out of view. If you cannot "
                  "see it, return dx_cm=0,dy_cm=0 with LOW confidence so the controller "
                  "widens its search.",
    F_OCCLUSION:  BASE_INSTR + " The object is partially occluded (often by the gripper). "
                  "Estimate the offset to the visible portion's centroid; lower confidence "
                  "if heavily occluded.",
    F_ORIENTATION: BASE_INSTR + " The object is mis-oriented. Give the planar offset to its "
                  "centroid; note orientation in reasoning.",
}
CLASSIFY_PROMPT = ("Classify this failed top-down grasp. Reply ONLY JSON: "
                   '{"failure_type":"POSITIONAL_OFFSET|NO_OBJECT|OCCLUSION|ORIENTATION",'
                   '"confidence":<0-1>}')

# (4) sim-to-real noise: base sigmas, scaled by an ablation multiplier
PERCEPTION_SIGMA_CM = 1.5       # VLM estimate noise per axis at multiplier 1.0
ACTUATION_SIGMA_CM = 0.8        # robot doesn't land exactly where commanded

# camera->base sign calibration (flip once if a correction worsens the miss)
CAM_DX_SIGN, CAM_DY_SIGN = +1.0, +1.0

try:
    import ollama
    _OLLAMA = True
except Exception:
    _OLLAMA = False


# ================================================================= MCS logging (5)
class MCSLogger:
    """Emits semantic, human-readable fault records (not binary codes) that a
    Master Control System could consume, as JSON-lines."""
    def __init__(self, path, node_id="panda_cell_01", verbose=False):
        """Create a new JSONL log file and configure optional console output."""
        self.path = path
        self.node_id = node_id
        self.verbose = verbose
        self.records = []
        open(path, "w").close()

    def emit(self, event, message, **fields):
        """Record one semantic MCS event with structured fields."""
        rec = {"ts": round(time.time(), 3), "node_id": self.node_id,
               "event": event, "semantic_message": message}
        rec.update(fields)
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.verbose:
            conf = f"  conf={fields['confidence']:.2f}" if "confidence" in fields else ""
            print(f"    [MCS] {event:16s} {message}{conf}")
        return rec


# ================================================================= sim + camera
def build_world():
    """Create a fresh PyBullet world with the floor and Franka Panda robot."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(PANDA_ARM_JOINTS, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in PANDA_FINGER_JOINTS:
        p.resetJointState(robot, j, FINGER_OPEN)
    return robot


def spawn_object(xy):
    """Place the orange cube at the requested XY location."""
    obj = p.loadURDF("cube_small.urdf", basePosition=[xy[0], xy[1], 0.02])
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    return obj


def ee_pose(robot):
    """Return the end-effector position and orientation."""
    s = p.getLinkState(robot, PANDA_EE_LINK, computeForwardKinematics=True)
    return np.array(s[4]), s[5]


def capture(robot, path, jitter=0.0):
    """Render an eye-in-hand frame path for the real VLM backend."""
    pos, orn = ee_pose(robot)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    fwd = rot @ np.array([0, 0, 1]); up = rot @ np.array([0, 1, 0])
    eye = pos + fwd * 0.05 + np.array([jitter, jitter, 0])
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
    p.getCameraImage(240, 240, view, proj, renderer=p.ER_TINY_RENDERER)
    return str(path)


def _move(robot, xyz, steps=60):
    """Move the gripper toward a Cartesian target using inverse kinematics."""
    jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), DOWN_ORN)
    for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for _ in range(steps):
        p.stepSimulation()


def attempt_grasp(robot, aim_xy, sabotage=None, act_sigma_cm=0.0):
    """Execute a grasp with optional sabotage and actuation noise."""
    gx, gy = aim_xy
    if sabotage is not None:
        gx += sabotage[0]; gy += sabotage[1]
    _move(robot, [gx, gy, 0.25]); _move(robot, [gx, gy, 0.05])
    # actuation noise: robot lands slightly off the commanded point
    land = np.array([gx, gy]) + np.random.normal(0, act_sigma_cm / 100.0, 2)
    return np.array([gx, gy]), land


def succeeded(land_xy, object_xy):
    """Check whether the landing point is inside the capture radius."""
    return float(np.linalg.norm(np.array(land_xy) - np.array(object_xy))) < CAPTURE_RADIUS


def camera_to_base(dx_cm, dy_cm):
    """Convert camera-frame centimetre correction into base-frame metres."""
    return np.array([CAM_DX_SIGN * dx_cm, CAM_DY_SIGN * dy_cm]) / 100.0


# ================================================================= VLM backends
class MockVLM:
    """Ground-truth + noise stand-in. Implements temporal aggregation, routed
    prompts (recorded), and a confidence model tied to noise/offset/agreement."""
    def __init__(self):
        """Initialize the mock call counter."""
        self.calls = 0

    def classify(self, gripper_xy, object_xy):
        """Classify the current miss using ground-truth offset size."""
        off_cm = np.linalg.norm(np.array(object_xy) - np.array(gripper_xy)) * 100.0
        if off_cm > OUT_OF_FRAME_CM:
            return F_NO_OBJECT, 0.9
        return F_POSITIONAL, 0.85

    def diagnose(self, gripper_xy, object_xy, failure_type, k, perc_sigma_cm):
        """Return a mock routed correction with temporal aggregation and confidence."""
        self.calls += 1
        t0 = time.perf_counter()
        _ = PROMPTS[failure_type]                      # routing exercised
        true = (np.array(object_xy) - np.array(gripper_xy)) * 100.0
        draws = [true + np.random.normal(0, perc_sigma_cm, 2) for _ in range(k)]
        est = np.median(np.array(draws), axis=0)       # (2) temporal aggregation
        spread = float(np.mean(np.std(np.array(draws), axis=0))) if k > 1 else perc_sigma_cm
        off_cm = float(np.linalg.norm(true))
        base = 0.95 - 0.09 * perc_sigma_cm - 0.02 * off_cm
        temporal_bonus = 0.07 * (k - 1) / (1.0 + spread)
        conf = float(np.clip(base + temporal_bonus + np.random.normal(0, 0.03), 0.05, 0.99))
        return {"reasoning": "object offset from centre",
                "dx_cm": float(est[0]), "dy_cm": float(est[1]),
                "confidence": conf, "_agreement": 1.0 / (1.0 + spread),
                "_frames": k, "_latency_s": time.perf_counter() - t0}


class RealVLM:
    """Real Llama 3.2 Vision via Ollama. Uses routed prompts and sends all K
    temporal frames in a single query."""
    def __init__(self, model=VLM_MODEL):
        """Verify Ollama is available and remember the model name."""
        if not _OLLAMA:
            raise RuntimeError("USE_REAL_VLM=True but 'ollama' not installed. "
                               "pip install ollama; install Ollama for Windows; "
                               "ollama pull llama3.2-vision.")
        self.model = model

    def _chat(self, system, image_paths):
        """Send a routed prompt and image paths to the Ollama vision model."""
        return ollama.chat(model=self.model, format="json",
                           messages=[{"role": "system", "content": system},
                                     {"role": "user",
                                      "content": "Return the JSON.",
                                      "images": image_paths}],
                           options={"temperature": 0.1})["message"]["content"]

    def classify(self, image_paths):
        """Ask the real VLM to classify the failure type."""
        try:
            d = json.loads(self._chat(CLASSIFY_PROMPT, image_paths))
            return d.get("failure_type", F_POSITIONAL), float(d.get("confidence", 0.5))
        except Exception:
            return F_POSITIONAL, 0.5

    def diagnose(self, image_paths, failure_type, **_):
        """Ask the real VLM for a correction using the routed failure prompt."""
        t0 = time.perf_counter()
        try:
            d = json.loads(self._chat(PROMPTS[failure_type], image_paths))  # routed
            return {"reasoning": d.get("reasoning", ""),
                    "dx_cm": float(d["dx_cm"]), "dy_cm": float(d["dy_cm"]),
                    "confidence": float(d.get("confidence", 0.5)),
                    "_frames": len(image_paths),
                    "_latency_s": time.perf_counter() - t0}
        except Exception as e:
            print(f"    [RealVLM] parse fail: {e}")
            return None


def make_vlm():
    """Select the real or mock VLM backend from the global flag."""
    return RealVLM() if USE_REAL_VLM else MockVLM()


# ================================================================= FSM episode
def run_episode(robot, object_xy, nominal_xy, sabotage, mode, vlm, mcs,
                perc_sigma_cm, act_sigma_cm, tag="", render=False):
    """mode in {'open','closed_single','closed_temporal'}.
    Returns dict(success, recoveries, outcome, vlm_time)."""
    k = TEMPORAL_FRAMES if mode == "closed_temporal" else 1
    aim, land = attempt_grasp(robot, nominal_xy, sabotage, act_sigma_cm)
    if succeeded(land, object_xy):
        return {"success": True, "recoveries": 0, "outcome": "first_try", "vlm_time": 0.0}

    if mode == "open":
        mcs.emit("GRASP_FAILURE",
                 "Open-loop grasp missed; no recovery configured (binary fault).",
                 failure_type="UNDIAGNOSED", resolved=False)
        return {"success": False, "recoveries": 0, "outcome": "open_fail", "vlm_time": 0.0}

    target = np.array(aim, dtype=float)
    vlm_time = 0.0
    for attempt in range(1, MAX_RECOVERIES + 1):
        _move(robot, [target[0], target[1], DIAGNOSE_HOVER_Z])   # hover to view

        # ---- (3) failure classification + prompt routing ----
        if USE_REAL_VLM:
            frames = [capture(robot, FRAMES_DIR / f"{tag}_{attempt}_{i}.png",
                              jitter=CAM_JITTER_M * i) for i in range(k)]
            ftype, fconf = vlm.classify(frames)
        else:
            if render:
                capture(robot, FRAMES_DIR / f"{tag}_{attempt}.png")
            ftype, fconf = vlm.classify(land, object_xy)
            frames = None

        # ---- NO_OBJECT routing: widen search back to nominal task pose ----
        if ftype == F_NO_OBJECT:
            mcs.emit("FAILURE_ROUTED",
                     "Object not in field of view; widening search to nominal task pose.",
                     failure_type=ftype, confidence=round(fconf, 2), attempt=attempt,
                     recovery_action="WIDEN_SEARCH", resolved=False)
            target = np.array(nominal_xy, dtype=float)          # re-acquire strategy
            _move(robot, [target[0], target[1], DIAGNOSE_HOVER_Z])
            if USE_REAL_VLM:
                frames = [capture(robot, FRAMES_DIR / f"{tag}_{attempt}_r.png")]
                ftype, fconf = vlm.classify(frames)
            else:
                ftype, fconf = vlm.classify(target, object_xy)
            if ftype == F_NO_OBJECT:
                mcs.emit("RECOVERY_ABORTED",
                         "Object could not be re-acquired after search; escalating to MCS.",
                         failure_type=F_NO_OBJECT, attempt=attempt, resolved=False)
                return {"success": False, "recoveries": attempt,
                        "outcome": "no_object_abort", "vlm_time": vlm_time}

        # ---- diagnose (routed prompt + temporal frames) ----
        if USE_REAL_VLM:
            diag = vlm.diagnose(frames, ftype)
        else:
            diag = vlm.diagnose(land if attempt == 1 else target, object_xy, ftype, k, perc_sigma_cm)
        if diag is None:
            continue
        vlm_time += diag["_latency_s"]
        conf = diag["confidence"]

        # ---- (1) confidence gating ----
        if conf < CONF_REQUERY:
            mcs.emit("RECOVERY_ABORTED",
                     f"Diagnosis confidence {conf:.2f} below safe threshold "
                     f"{CONF_REQUERY:.2f}; aborting to avoid unsafe motion.",
                     failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                     recovery_action="ABORT_LOW_CONFIDENCE", resolved=False)
            return {"success": False, "recoveries": attempt,
                    "outcome": "low_conf_abort", "vlm_time": vlm_time}

        if conf < CONF_ACT:
            # marginal: re-query with the full temporal window before acting
            kk = max(k, TEMPORAL_FRAMES)
            if USE_REAL_VLM:
                more = frames + [capture(robot, FRAMES_DIR / f"{tag}_{attempt}_m{i}.png",
                                         jitter=CAM_JITTER_M * (i + k)) for i in range(kk - len(frames))]
                diag2 = vlm.diagnose(more, ftype)
            else:
                diag2 = vlm.diagnose(land if attempt == 1 else target, object_xy, ftype, kk, perc_sigma_cm)
            if diag2 is not None:
                vlm_time += diag2["_latency_s"]
                mcs.emit("DIAGNOSIS_REQUERIED",
                         f"Marginal confidence {conf:.2f}; re-queried with {kk} frames "
                         f"-> {diag2['confidence']:.2f}.",
                         failure_type=ftype, confidence=round(diag2["confidence"], 2),
                         attempt=attempt, recovery_action="REQUERY")
                diag, conf = diag2, diag2["confidence"]
                if conf < CONF_REQUERY:
                    mcs.emit("RECOVERY_ABORTED",
                             "Confidence still too low after re-query; aborting.",
                             failure_type=ftype, confidence=round(conf, 2),
                             attempt=attempt, resolved=False)
                    return {"success": False, "recoveries": attempt,
                            "outcome": "low_conf_abort", "vlm_time": vlm_time}

        # ---- act: apply routed correction and retry ----
        corr = camera_to_base(diag["dx_cm"], diag["dy_cm"])
        target = target + corr
        mag = float(np.hypot(diag["dx_cm"], diag["dy_cm"]))
        mcs.emit("CORRECTION_APPLIED",
                 f"{ftype}: object {mag:.1f}cm from target; applying VLM correction "
                 f"(dx={diag['dx_cm']:+.1f}, dy={diag['dy_cm']:+.1f}) cm and retrying.",
                 failure_type=ftype, confidence=round(conf, 2),
                 correction_cm={"dx": round(diag["dx_cm"], 2), "dy": round(diag["dy_cm"], 2)},
                 frames_used=diag.get("_frames", 1), attempt=attempt,
                 recovery_action="RETRY", resolved=False)

        aim2, land = attempt_grasp(robot, tuple(target), None, act_sigma_cm)
        if succeeded(land, object_xy):
            mcs.emit("RECOVERY_SUCCEEDED",
                     f"Grasp recovered on attempt {attempt}.",
                     failure_type=ftype, confidence=round(conf, 2),
                     attempt=attempt, resolved=True)
            return {"success": True, "recoveries": attempt,
                    "outcome": "recovered", "vlm_time": vlm_time}

    mcs.emit("RECOVERY_EXHAUSTED",
             f"Recovery unsuccessful after {MAX_RECOVERIES} attempts; escalating to MCS.",
             failure_type=F_POSITIONAL, resolved=False)
    return {"success": False, "recoveries": MAX_RECOVERIES,
            "outcome": "exhausted", "vlm_time": vlm_time}


# ================================================================= drivers
def qualitative_trace(vlm):
    """A few narrated episodes that exercise every feature, with verbose MCS logs."""
    print("=" * 74)
    print("QUALITATIVE TRACE  (confidence gating, routing, temporal, MCS logging)")
    print("=" * 74)
    mcs = MCSLogger(OUT / "mcs_faults.jsonl", verbose=True)
    # (object_xy, nominal_xy, sabotage, mode, perc_sigma, act_sigma, label)
    cases = [
        (BASE_XY, BASE_XY, (0.038, -0.022), "closed_temporal", 1.2, 0.6,
         "positional miss -> temporal diagnose -> recover"),
        (BASE_XY, BASE_XY, (0.030,  0.030), "closed_temporal", 3.6, 0.8,
         "high perception noise -> marginal confidence -> re-query"),
        (BASE_XY, BASE_XY, (0.028, -0.030), "closed_single", 6.0, 1.0,
         "very noisy single-frame -> low confidence -> safe abort"),
        (BASE_XY, BASE_XY, (0.150,  0.020), "closed_temporal", 1.2, 0.6,
         "gross offset -> object out of view -> NO_OBJECT routing -> widen search"),
    ]
    for i, (obj, nom, sab, mode, ps, as_, label) in enumerate(cases, 1):
        print(f"\n-- Case {i}: {label}")
        robot = build_world(); spawn_object(obj)
        for _ in range(40):
            p.stepSimulation()
        r = run_episode(robot, obj, nom, sab, mode, vlm, mcs, ps, as_,
                        tag=f"q{i}", render=True)
        print(f"   outcome={r['outcome']:16s} success={r['success']} "
              f"recoveries={r['recoveries']}")
    print(f"\n  wrote semantic fault log -> mcs_faults.jsonl "
          f"({len(mcs.records)} records)")
    return mcs


def ablation(vlm):
    """(4) sim-to-real noise ablation: success vs noise for three conditions."""
    print("\n" + "=" * 74)
    print("SIM-TO-REAL NOISE ABLATION")
    print("=" * 74)
    mults = [0.0, 1.0, 2.0, 3.0]
    conditions = ["open", "closed_single", "closed_temporal"]
    N = 35
    quiet = MCSLogger(FRAMES_DIR / "_ablation_mcs.jsonl")  # not verbose
    results = {c: [] for c in conditions}

    for m in mults:
        ps, as_ = PERCEPTION_SIGMA_CM * m, ACTUATION_SIGMA_CM * m
        line = f"  noise x{m:>3.1f} (perc {ps:.1f}cm, act {as_:.1f}cm): "
        for c in conditions:
            succ = 0
            sab_rng = random.Random(1000)             # identical saboteur per condition
            for t in range(N):
                robot = build_world(); spawn_object(BASE_XY)
                sab = (sab_rng.uniform(-0.045, 0.045), sab_rng.uniform(-0.045, 0.045))
                r = run_episode(robot, BASE_XY, BASE_XY, sab, c, vlm, quiet, ps, as_,
                                tag=f"abl", render=False)
                succ += int(r["success"])
            rate = 100 * succ / N
            results[c].append(rate)
            line += f"{c.split('_')[-1][:4]}={rate:3.0f}%  "
        print(line)

    # chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    styles = {"open": ("#c0563b", "o", "Open-loop (baseline)"),
              "closed_single": ("#d69b3b", "s", "Closed-loop, single-frame"),
              "closed_temporal": ("#3b7dc0", "^", f"Closed-loop, {TEMPORAL_FRAMES}-frame temporal")}
    for c in conditions:
        col, mk, lab = styles[c]
        ax.plot(mults, results[c], marker=mk, color=col, linewidth=2, markersize=8, label=lab)
    ax.set_xlabel("Sim-to-real noise multiplier  (perception + actuation)")
    ax.set_ylabel("Grasp success rate (%)")
    ax.set_title("Sim-to-real noise ablation: recovery robustness")
    ax.set_ylim(0, 100); ax.set_xticks(mults); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "sim2real_ablation.png", dpi=130)
    print(f"\n  wrote ablation chart -> sim2real_ablation.png")
    return results


def main():
    """Run the advanced qualitative trace and sim-to-real ablation."""
    global DOWN_ORN
    p.connect(p.DIRECT)
    DOWN_ORN = p.getQuaternionFromEuler([np.pi, 0, 0])
    np.random.seed(0); random.seed(0)
    vlm = make_vlm()
    print(f"VLM backend: {'RealVLM (Llama 3.2 Vision)' if USE_REAL_VLM else 'MockVLM'}\n")
    qualitative_trace(vlm)
    ablation(vlm)
    p.disconnect()


if __name__ == "__main__":
    main()
