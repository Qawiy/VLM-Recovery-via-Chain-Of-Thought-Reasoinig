"""
================================================================================
 AUTONOMOUS FAILURE RECOVERY IN ROBOTIC MANIPULATION USING VISION-LANGUAGE MODELS
 ------------------------------------------------------------------------------
 ALL-IN-ONE SCRIPT  --  the entire project's Python in a single file.
================================================================================

 This file merges every script in the project into one, organised into clearly
 labelled sections. Each experiment is dispatched from the command line:

     python project_all_in_one.py base       # <- from demo.py
     python project_all_in_one.py advanced    # <- from demo_advanced.py
     python project_all_in_one.py trials       # <- from demo_trials.py   (--n 120)
     python project_all_in_one.py probe         # <- from probe_vlm.py
     python project_all_in_one.py video          # <- from render_*.py
     python project_all_in_one.py all             # base + advanced + trials

 WHAT EACH ORIGINAL FILE CONTRIBUTED (highlighted per section below):
   demo.py            -> SECTION 6  : base open-vs-closed recovery experiment
   demo_advanced.py   -> SECTIONS 3,4,5,7 : the 5 novel extensions + ablation
   demo_trials.py     -> SECTIONS 4,8  : 3-DOF location+orientation study + Wilson CI
   probe_vlm.py       -> SECTION 9  : real-model single-frame calibration tool
   render_*.py        -> SECTION 10 : narrated video generation

 The five novel contributions all live in the SHARED CORE (Sections 3-5) and are
 switched on per-experiment by flags:
   (1) confidence-gated FSM transitions        -> run_episode(use_gate=...)
   (2) temporal multi-frame VLM queries         -> run_episode(temporal_frames=...)
   (3) failure-type-conditioned prompt routing   -> run_episode(use_routing=...)
   (4) sim-to-real noise ablation                 -> experiment_advanced()
   (5) semantic MCS fault logging                  -> MCSLogger + run_episode(mcs=...)

 Everything runs headless on the MOCK VLM by default (no GPU needed). Set
 USE_REAL_VLM=True to route diagnoses to the real Llama 3.2 Vision model via Ollama.
================================================================================
"""

import os, sys, csv, math, json, time, random, argparse
import numpy as np
import pybullet as p
import pybullet_data

# All generated files (charts, videos, CSV, fault log, camera frames) are saved
# into an "outputs" folder next to this script, on any OS. Uses the sandbox path
# only if it already exists; otherwise saves locally beside project_all_in_one.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)
FRAMES_DIR = os.path.join(OUT, "_frames"); os.makedirs(FRAMES_DIR, exist_ok=True)



# ==============================================================================
# SECTION 1 -- CONFIGURATION  (all tunable constants in one place)
#   Highlights: the knobs that change robot, task tolerances, noise, the
#   confidence gate, temporal frames, and the mock/real VLM switch.
# ==============================================================================

# --- VLM backend switch (one flag flips the whole project to the real model) ---
USE_REAL_VLM = True
# Vision model served by Ollama. qwen2.5vl is used instead of llama3.2-vision
# because current Ollama engines (v0.30+) dropped the 'mllama' architecture that
# Llama 3.2 Vision needs, which raises "unknown model architecture: 'mllama'".
# qwen2.5vl is a drop-in vision model that loads on current and older Ollama.
# To use it once:  ollama pull qwen2.5vl
VLM_MODEL = "qwen2.5vl"

# --- Franka Panda constants (indices into PyBullet's bundled panda.urdf) ---
PANDA_EE_LINK = 11
PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]
FINGER_OPEN = 0.04

# --- task success tolerances (the "capture model") ---
CAPTURE_RADIUS = 0.028          # m   position tolerance
YAW_TOL_DEG = 12.0              # deg orientation tolerance (3-DOF trials only)

# --- finite-state-machine settings ---
MAX_RECOVERIES = 3
CONF_ACT = 0.60                 # (1) confidence gate: >= act
CONF_REQUERY = 0.35             # (1) confidence gate: in-between -> re-query; below -> abort
TEMPORAL_FRAMES = 3             # (2) frames aggregated per diagnosis
OUT_OF_FRAME_CM = 12.0          # (3) beyond this, object is "not in view" (NO_OBJECT)

# --- camera->base sign calibration (flip once if a correction worsens the miss) ---
CAM_DX_SIGN, CAM_DY_SIGN, CAM_DTH_SIGN = +1.0, +1.0, +1.0

# --- failure taxonomy (3) ---
F_POS, F_ORI, F_NO = "POSITIONAL_OFFSET", "ORIENTATION", "NO_OBJECT"

# --- per-experiment noise defaults ---
BASE_PERC_POS = 2.0                         # demo.py mock perception noise
ABL_PERC, ABL_ACT = 1.5, 0.8                # demo_advanced.py ablation base sigmas
TRIALS_PERC_POS, TRIALS_PERC_YAW = 2.3, 7.5 # demo_trials.py perception noise
TRIALS_ACT_POS, TRIALS_ACT_YAW = 1.1, 3.5   # demo_trials.py actuation noise

# --- workspace (3-DOF trials) ---
X_RANGE, Y_RANGE, YAW_RANGE_DEG = (0.40, 0.60), (-0.20, 0.20), (-40, 40)
SAB_POS, SAB_YAW_DEG = 0.045, 25.0
BASE_XY = (0.50, 0.0)

_DOWN = None  # set after PyBullet connects

# --- routed, failure-type-specific prompts for the REAL model (3) ---
# The model REASONS about the object's location and orientation first (chain-of-
# thought), then derives the correction. The `reasoning` field is kept and logged.
_BASE_INSTR = ("You are the visual reasoning module of a robot arm performing a "
               "top-down pick-and-place. You see a downward eye-in-hand camera image. "
               "The image centre (crosshair) is where the gripper is currently aiming. "
               "Camera axes: +x = image right, +y = image up. The orange cube is the "
               "target object. "
               "Think step by step before answering: "
               "(1) LOCATE the cube -- describe where it is relative to the crosshair "
               "(left/right, up/down) and estimate how far off in centimetres; "
               "(2) ORIENT -- estimate the cube's rotation (yaw) in degrees relative to "
               "the gripper; "
               "(3) DECIDE the correction (dx_cm, dy_cm, dtheta_deg) to ADD to the "
               "gripper's aim so it lands on and aligns with the cube. "
               "Put that step-by-step reasoning in the \"reasoning\" field, then give the "
               "numbers. Respond with ONE JSON object and nothing else -- no markdown, "
               "no code fences. Exact keys and value types: "
               '{"reasoning": "short text: where the cube is and how it is rotated", '
               '"dx_cm": number, "dy_cm": number, "dtheta_deg": number, '
               '"confidence": number between 0 and 1}.')
PROMPTS = {
    F_POS: _BASE_INSTR + " The object is visible but off-centre.",
    F_ORI: _BASE_INSTR + " The object is mainly mis-rotated; estimate dtheta_deg.",
    F_NO:  _BASE_INSTR + " The object may be out of view; if unseen return zeros with LOW confidence.",
}
CLASSIFY_PROMPT = (
    "You view a downward eye-in-hand camera image after a failed top-down grasp. "
    "Classify the failure. Respond with ONE JSON object and nothing else -- no prose, "
    "no markdown, no code fences. failure_type must be exactly one of "
    '"POSITIONAL_OFFSET" (cube visible but off the centre crosshair), '
    '"ORIENTATION" (cube centred but clearly rotated), or '
    '"NO_OBJECT" (no cube visible). '
    'Format: {"failure_type": "POSITIONAL_OFFSET", "confidence": 0.0}.')

try:
    import ollama
    _OLLAMA = True
except Exception:
    _OLLAMA = False


# ==============================================================================
# SECTION 2 -- SIMULATION CORE  (world, robot, camera, grasp primitive)
#   Highlights (shared by demo.py / demo_advanced.py / demo_trials.py):
#   building the scene, the eye-in-hand RGB-D camera, inverse-kinematics motion,
#   and the top-down grasp primitive + the SABOTEUR that induces failures.
# ==============================================================================

def yaw_to_quat(yaw):
    """Gripper-pointing-down orientation, rotated by `yaw` about vertical."""
    return p.getQuaternionFromEuler([math.pi, 0, yaw])


def build_world(obj_xy=BASE_XY, obj_yaw=0.0):
    """Reset sim; load floor, Panda (rest pose), and the orange cube at (xy, yaw)."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(); p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(PANDA_ARM_JOINTS, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in PANDA_FINGER_JOINTS:
        p.resetJointState(robot, j, FINGER_OPEN)
    obj = p.loadURDF("cube_small.urdf", [obj_xy[0], obj_xy[1], 0.02],
                     p.getQuaternionFromEuler([0, 0, obj_yaw]))
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])   # high contrast for real VLM
    for _ in range(40):
        p.stepSimulation()
    return robot, obj


def ee_pose(robot):
    s = p.getLinkState(robot, PANDA_EE_LINK, computeForwardKinematics=True)
    return np.array(s[4]), s[5]


def capture_rgbd(robot, save_path=None, jitter=0.0):
    """Render the eye-in-hand view looking down from the gripper (RGB + depth)."""
    pos, orn = ee_pose(robot)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    fwd, up = rot @ np.array([0, 0, 1]), rot @ np.array([0, 1, 0])
    eye = pos + fwd * 0.05 + np.array([jitter, jitter, 0])
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
    _, _, rgba, _, _ = p.getCameraImage(320, 320, view, proj, renderer=p.ER_TINY_RENDERER)
    if save_path:
        from PIL import Image
        Image.fromarray(np.reshape(rgba, (320, 320, 4))[:, :, :3].astype("uint8")).save(save_path)
    return save_path


def move(robot, xyz, yaw=0.0, steps=55):
    """Inverse-kinematics drive: put the gripper at xyz with the given yaw."""
    jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), yaw_to_quat(yaw))
    for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for _ in range(steps):
        p.stepSimulation()


def attempt_grasp(robot, aim_xy, aim_yaw=0.0, sabotage=None,
                  act_pos_cm=0.0, act_yaw_deg=0.0):
    """Top-down grasp primitive: hover -> descend. `sabotage`=(dx,dy[,dyaw]) is the
    injected error that makes the grasp MISS. Actuation noise models a robot that
    doesn't land exactly where commanded. Returns (aimed pose, actual landing)."""
    x, y, yaw = aim_xy[0], aim_xy[1], aim_yaw
    if sabotage is not None:
        x += sabotage[0]; y += sabotage[1]
        if len(sabotage) > 2:
            yaw += sabotage[2]
    move(robot, [x, y, 0.25], yaw); move(robot, [x, y, 0.05], yaw)
    land_xy = np.array([x, y]) + np.random.normal(0, act_pos_cm / 100.0, 2)
    land_yaw = yaw + (math.radians(np.random.normal(0, act_yaw_deg)) if act_yaw_deg > 0 else 0.0)
    return (np.array([x, y]), yaw), (land_xy, land_yaw)


def succeeded(land, obj_xy, obj_yaw=0.0, use_orientation=False):
    """Capture model: success if within position tolerance (and yaw tolerance in 3-DOF)."""
    land_xy, land_yaw = land
    pos_ok = float(np.linalg.norm(land_xy - np.array(obj_xy))) < CAPTURE_RADIUS
    if not use_orientation:
        return pos_ok
    return pos_ok and abs(math.degrees(land_yaw - obj_yaw)) < YAW_TOL_DEG


def classify_state(aim_xy, aim_yaw, obj_xy, obj_yaw, use_orientation):
    """(3) Ground-truth failure classification used by the MOCK VLM."""
    pos_cm = float(np.linalg.norm(np.array(obj_xy) - np.array(aim_xy))) * 100
    if pos_cm > OUT_OF_FRAME_CM:
        return F_NO
    if use_orientation and abs(math.degrees(obj_yaw - aim_yaw)) > 10 and pos_cm < 3.0:
        return F_ORI
    return F_POS


def camera_to_base(dx_cm, dy_cm):
    """Convert the VLM's camera-frame correction (cm) to a base-frame move (m)."""
    return np.array([CAM_DX_SIGN * dx_cm, CAM_DY_SIGN * dy_cm]) / 100.0


# ==============================================================================
# SECTION 3 -- VLM BACKENDS  (mock vs real; same interface)
#   Highlights: the drop-in pair that makes USE_REAL_VLM a one-line switch.
#   MockVLM = ground truth + noise (contributions 2 & 3 baked in: temporal median
#   aggregation + confidence model). RealVLM = live Llama 3.2 Vision via Ollama.
# ==============================================================================

class MockVLM:
    """Emulates Llama 3.2 Vision from ground truth + noise. Returns dx_cm, dy_cm,
    dtheta_deg, confidence -- identical shape to RealVLM."""
    def classify(self, aim_xy, aim_yaw, obj_xy, obj_yaw, use_orientation, frames=None):
        return classify_state(aim_xy, aim_yaw, obj_xy, obj_yaw, use_orientation), 0.85

    def diagnose(self, ftype, aim_xy, aim_yaw, obj_xy, obj_yaw, k, use_orientation,
                 perc_pos_cm, perc_yaw_deg, frames=None):
        t0 = time.perf_counter()
        true_pos = (np.array(obj_xy) - np.array(aim_xy)) * 100.0
        true_yaw = math.degrees(obj_yaw - aim_yaw) if use_orientation else 0.0
        pos_draws = [true_pos + np.random.normal(0, perc_pos_cm, 2) for _ in range(k)]  # (2)
        est_pos = np.median(np.array(pos_draws), axis=0)
        if use_orientation:
            est_yaw = float(np.median([true_yaw + np.random.normal(0, perc_yaw_deg) for _ in range(k)]))
        else:
            est_yaw = 0.0
        spread = float(np.mean(np.std(np.array(pos_draws), axis=0))) if k > 1 else perc_pos_cm
        off = float(np.linalg.norm(true_pos))
        conf = float(np.clip(0.95 - 0.09 * perc_pos_cm - 0.02 * off
                             + 0.07 * (k - 1) / (1 + spread) + np.random.normal(0, 0.02),
                             0.05, 0.99))
        # synthesize a human-readable "reasoning" so the mock matches RealVLM's shape
        lr = "right" if est_pos[0] > 0 else "left"
        ud = "up" if est_pos[1] > 0 else "down"
        reasoning = (f"cube is ~{off:.1f} cm from the crosshair ({lr}/{ud})"
                     + (f", rotated ~{est_yaw:+.0f} deg" if use_orientation else "")
                     + f"; correcting dx={est_pos[0]:+.1f}, dy={est_pos[1]:+.1f} cm.")
        return {"dx_cm": float(est_pos[0]), "dy_cm": float(est_pos[1]),
                "dtheta_deg": est_yaw, "confidence": conf, "reasoning": reasoning,
                "_frames": k, "_latency_s": time.perf_counter() - t0}


def _extract_json(text):
    """Robustly pull a JSON object out of a model reply. Tolerates code fences and
    stray prose that a vision model may add even when JSON output is requested."""
    if not text:
        raise ValueError("empty model reply")
    t = text.strip()
    if "```" in t:                      # strip ```json ... ``` fences
        t = t.split("```")[1] if t.count("```") >= 2 else t.replace("```", "")
        t = t[4:].strip() if t.lower().startswith("json") else t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]            # keep only the first {...} block
    return json.loads(t)


class RealVLM:
    """Live vision model (qwen2.5vl by default) via Ollama. Uses routed prompts and
    sends all temporal frames in one query. Coordinate args are ignored (it only
    sees the images)."""
    def __init__(self, model=VLM_MODEL):
        if not _OLLAMA:
            raise RuntimeError("USE_REAL_VLM=True but 'ollama' not installed. Run: "
                               "pip install ollama  (and install the Ollama app), then: "
                               f"ollama pull {VLM_MODEL}")
        self.model = model
        self._preflight()

    def _preflight(self):
        """Fail fast with a clear message if the model can't be loaded."""
        try:
            ollama.chat(model=self.model,
                        messages=[{"role": "user", "content": "reply with the word ok"}],
                        options={"num_predict": 1})
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "no such model" in msg or "try pulling" in msg:
                raise RuntimeError(
                    f"Ollama does not have '{self.model}'. Download it once with:\n"
                    f"    ollama pull {self.model}\n"
                    "Then run again. To run without the real model, set USE_REAL_VLM=False."
                ) from e
            if "mllama" in msg:
                raise RuntimeError(
                    "This Ollama build cannot load Llama 3.2 Vision's 'mllama' architecture. "
                    f"This project now defaults to '{self.model}', which avoids that. "
                    f"Run:  ollama pull {self.model}"
                ) from e
            raise RuntimeError(
                f"Could not load model '{self.model}' via Ollama: {e}\n"
                f"Try:  ollama pull {self.model}   (or set USE_REAL_VLM=False)."
            ) from e

    def _chat(self, system, images):
        return ollama.chat(model=self.model, format="json",
                           messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": "Return the JSON.",
                                      "images": images}],
                           options={"temperature": 0.1})["message"]["content"]

    def classify(self, aim_xy, aim_yaw, obj_xy, obj_yaw, use_orientation, frames=None):
        try:
            d = _extract_json(self._chat(CLASSIFY_PROMPT, frames or []))
            ft = str(d.get("failure_type", F_POS)).upper()
            ft = ft if ft in (F_POS, F_ORI, F_NO) else F_POS
            return ft, float(d.get("confidence", 0.5))
        except Exception:
            return F_POS, 0.5

    def diagnose(self, ftype, aim_xy, aim_yaw, obj_xy, obj_yaw, k, use_orientation,
                 perc_pos_cm, perc_yaw_deg, frames=None):
        t0 = time.perf_counter()
        try:
            d = _extract_json(self._chat(PROMPTS[ftype], frames or []))
            return {"dx_cm": float(d["dx_cm"]), "dy_cm": float(d["dy_cm"]),
                    "dtheta_deg": float(d.get("dtheta_deg", 0.0)),
                    "confidence": float(d.get("confidence", 0.5)),
                    "reasoning": str(d.get("reasoning", "")).strip(),
                    "_frames": len(frames or []), "_latency_s": time.perf_counter() - t0}
        except Exception as e:
            print("    [RealVLM] parse fail:", e)
            return None


def make_vlm():
    """The single switch: mock (anywhere) vs real vision model (qwen2.5vl)."""
    return RealVLM() if USE_REAL_VLM else MockVLM()


# ==============================================================================
# SECTION 4 -- SEMANTIC MCS FAULT LOGGER  (novel contribution 5)
#   Highlights: writes human-readable, structured fault records (JSON-lines) a
#   factory Master Control System could consume -- not opaque binary codes.
# ==============================================================================

class MCSLogger:
    def __init__(self, path, node_id="panda_cell_01", verbose=False):
        self.path, self.node_id, self.verbose, self.records = path, node_id, verbose, []
        open(path, "w").close()

    def emit(self, event, message, **fields):
        rec = {"ts": round(time.time(), 3), "node_id": self.node_id,
               "event": event, "semantic_message": message, **fields}
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.verbose:
            c = f"  conf={fields['confidence']:.2f}" if "confidence" in fields else ""
            print(f"    [MCS] {event:18s} {message}{c}")
        return rec


# ==============================================================================
# SECTION 5 -- THE FINITE STATE MACHINE  (the closed loop; contributions 1,2,3)
#   Highlights: one unified run_episode drives every experiment. Feature flags
#   switch on the confidence gate (1), temporal frames (2), failure routing (3),
#   actuation noise, 2-DOF vs 3-DOF, and MCS logging (5).
# ==============================================================================

def run_episode(robot, obj_xy, obj_yaw, sabotage, vlm, *, closed, use_orientation=False,
                temporal_frames=1, use_gate=False, use_routing=False,
                perc_pos_cm=2.0, perc_yaw_deg=0.0, act_pos_cm=0.0, act_yaw_deg=0.0,
                mcs=None, tag=""):
    def _cap_frames(k):
        return [capture_rgbd(robot, os.path.join(FRAMES_DIR, f"{tag}_{k}_{i}.png"),
                             jitter=0.004 * i) for i in range(k)] if USE_REAL_VLM else None

    (aim_xy, aim_yaw), land = attempt_grasp(robot, obj_xy, obj_yaw, sabotage,
                                            act_pos_cm, act_yaw_deg)
    if succeeded(land, obj_xy, obj_yaw, use_orientation):
        return {"success": True, "recoveries": 0, "ftype": "-", "aborted": False}

    if not closed:
        if mcs:
            mcs.emit("GRASP_FAILURE", "Open-loop miss; no recovery configured (binary fault).",
                     failure_type="UNDIAGNOSED", resolved=False)
        return {"success": False, "recoveries": 0, "ftype": "-", "aborted": False}

    tgt_xy, tgt_yaw = np.array(aim_xy, float), aim_yaw
    ftype_seen = F_POS
    for attempt in range(1, MAX_RECOVERIES + 1):
        move(robot, [tgt_xy[0], tgt_xy[1], 0.22], tgt_yaw)       # hover to view
        frames = _cap_frames(temporal_frames)

        # (3) failure classification + prompt routing
        if use_routing:
            ftype, fconf = vlm.classify(tgt_xy, tgt_yaw, obj_xy, obj_yaw, use_orientation, frames)
        else:
            ftype, fconf = F_POS, 0.85
        ftype_seen = ftype

        # (3) NO_OBJECT -> widen search back to nominal task pose
        if ftype == F_NO:
            if mcs:
                mcs.emit("FAILURE_ROUTED", "Object not in view; widening search to nominal pose.",
                         failure_type=ftype, attempt=attempt, recovery_action="WIDEN_SEARCH",
                         resolved=False)
            tgt_xy = np.array(obj_xy, float)
            move(robot, [tgt_xy[0], tgt_xy[1], 0.22], tgt_yaw)
            frames = _cap_frames(temporal_frames)
            ftype = classify_state(tgt_xy, tgt_yaw, obj_xy, obj_yaw, use_orientation) \
                if not USE_REAL_VLM else vlm.classify(tgt_xy, tgt_yaw, obj_xy, obj_yaw,
                                                      use_orientation, frames)[0]
            if ftype == F_NO:
                if mcs:
                    mcs.emit("RECOVERY_ABORTED", "Could not re-acquire object; escalating.",
                             failure_type=F_NO, attempt=attempt, resolved=False)
                return {"success": False, "recoveries": attempt, "ftype": F_NO, "aborted": True}

        # (2) temporal multi-frame diagnosis, using the routed prompt
        diag = vlm.diagnose(ftype, tgt_xy, tgt_yaw, obj_xy, obj_yaw, temporal_frames,
                            use_orientation, perc_pos_cm, perc_yaw_deg, frames)
        if diag is None:
            continue
        conf = diag["confidence"]

        # (1) confidence gate
        if use_gate and conf < CONF_REQUERY:
            if mcs:
                mcs.emit("RECOVERY_ABORTED",
                         f"Confidence {conf:.2f} below safe threshold; aborting to avoid unsafe move.",
                         failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                         recovery_action="ABORT_LOW_CONFIDENCE", resolved=False)
            return {"success": False, "recoveries": attempt, "ftype": ftype_seen, "aborted": True}
        if use_gate and conf < CONF_ACT:
            frames2 = _cap_frames(max(temporal_frames, TEMPORAL_FRAMES))
            diag2 = vlm.diagnose(ftype, tgt_xy, tgt_yaw, obj_xy, obj_yaw,
                                 max(temporal_frames, TEMPORAL_FRAMES), use_orientation,
                                 perc_pos_cm, perc_yaw_deg, frames2 or frames)
            if diag2:
                if mcs:
                    mcs.emit("DIAGNOSIS_REQUERIED",
                             f"Marginal confidence {conf:.2f}; re-queried -> {diag2['confidence']:.2f}.",
                             failure_type=ftype, confidence=round(diag2["confidence"], 2),
                             attempt=attempt, recovery_action="REQUERY")
                diag, conf = diag2, diag2["confidence"]
                if conf < CONF_REQUERY:
                    if mcs:
                        mcs.emit("RECOVERY_ABORTED", "Still low confidence after re-query; aborting.",
                                 failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                                 resolved=False)
                    return {"success": False, "recoveries": attempt, "ftype": ftype_seen, "aborted": True}

        # act: apply the (routed) correction and retry
        tgt_xy = tgt_xy + camera_to_base(diag["dx_cm"], diag["dy_cm"])
        if use_orientation:
            tgt_yaw = tgt_yaw + math.radians(CAM_DTH_SIGN * diag["dtheta_deg"])
        if mcs:
            mag = float(np.hypot(diag["dx_cm"], diag["dy_cm"]))
            reasoning = diag.get("reasoning", "")
            mcs.emit("CORRECTION_APPLIED",
                     f"{ftype}: object {mag:.1f}cm off; applying correction "
                     f"(dx={diag['dx_cm']:+.1f}, dy={diag['dy_cm']:+.1f}) cm and retrying."
                     + (f" [VLM: {reasoning}]" if reasoning else ""),
                     failure_type=ftype, confidence=round(conf, 2),
                     vlm_reasoning=reasoning,
                     correction_cm={"dx": round(diag["dx_cm"], 2), "dy": round(diag["dy_cm"], 2)},
                     dtheta_deg=round(diag.get("dtheta_deg", 0.0), 1),
                     frames_used=diag.get("_frames", 1), attempt=attempt,
                     recovery_action="RETRY", resolved=False)

        _, land = attempt_grasp(robot, tuple(tgt_xy), tgt_yaw, None, act_pos_cm, act_yaw_deg)
        if succeeded(land, obj_xy, obj_yaw, use_orientation):
            if mcs:
                mcs.emit("RECOVERY_SUCCEEDED", f"Grasp recovered on attempt {attempt}.",
                         failure_type=ftype, attempt=attempt, resolved=True)
            return {"success": True, "recoveries": attempt, "ftype": ftype_seen, "aborted": False}

    if mcs:
        mcs.emit("RECOVERY_EXHAUSTED", f"Unrecovered after {MAX_RECOVERIES} attempts; escalating.",
                 failure_type=ftype_seen, resolved=False)
    return {"success": False, "recoveries": MAX_RECOVERIES, "ftype": ftype_seen, "aborted": False}


def wilson(k, n, z=1.96):
    """95% Wilson score confidence interval for a binomial proportion -> (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    ph = k / n
    denom = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return ph, max(0.0, c - half), min(1.0, c + half)


def _bar_chart(path, labels, rates, cis=None, title="", colors=("#c0563b", "#3b7dc0")):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(labels, rates, color=colors[:len(labels)], width=0.55)
    if cis:
        yerr = [[r - lo for r, (lo, hi) in zip(rates, cis)],
                [hi - r for r, (lo, hi) in zip(rates, cis)]]
        ax.errorbar(range(len(labels)), rates, yerr=yerr, fmt="none",
                    ecolor="black", capsize=7, elinewidth=1.5)
    for i, v in enumerate(rates):
        ax.text(i, min(v + 3, 100), f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("success rate (%)"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ==============================================================================
# SECTION 6 -- EXPERIMENT: BASE  (from demo.py)
#   Highlights: the core open-loop-vs-closed-loop result (position only, 2-DOF).
# ==============================================================================

def experiment_base(n=30):
    print("=" * 70 + "\nBASE EXPERIMENT (from demo.py): open-loop vs closed-loop\n" + "=" * 70)
    vlm = make_vlm(); summary = {}
    for cond, closed in [("open_loop", False), ("closed_loop", True)]:
        rng = random.Random(42); succ = 0
        for i in range(n):
            robot, _ = build_world(BASE_XY, 0.0)
            sab = (rng.uniform(-0.045, 0.045), rng.uniform(-0.045, 0.045))
            r = run_episode(robot, BASE_XY, 0.0, sab, vlm, closed=closed,
                            temporal_frames=1, perc_pos_cm=BASE_PERC_POS, tag=f"base{i}")
            succ += int(r["success"])
        rate = 100 * succ / n; summary[cond] = rate
        print(f"  {cond:12s}: {succ}/{n} = {rate:.0f}%")
    _bar_chart(os.path.join(OUT, "results_open_vs_closed.png"),
               ["Open-loop", "Closed-loop"], [summary["open_loop"], summary["closed_loop"]],
               title=f"Base: open vs closed ({n} trials)")
    print("  wrote results_open_vs_closed.png")


# ==============================================================================
# SECTION 7 -- EXPERIMENT: ADVANCED  (from demo_advanced.py)
#   Highlights: qualitative trace exercising the confidence gate (1), temporal
#   frames (2), routing (3) + widen-search, and MCS logging (5); then the
#   sim-to-real noise ablation (4).
# ==============================================================================

def experiment_advanced(ablation_n=25):
    print("=" * 70 + "\nADVANCED (from demo_advanced.py): 5 extensions + ablation\n" + "=" * 70)
    vlm = make_vlm()
    mcs = MCSLogger(os.path.join(OUT, "mcs_faults.jsonl"), verbose=True)

    print("\n-- qualitative trace --")
    cases = [  # (sabotage, temporal, perc_pos, act_pos, label)
        ((0.038, -0.022), 3, 1.2, 0.6, "positional -> temporal -> ACT -> recover"),
        ((0.030,  0.030), 3, 3.6, 0.8, "high noise -> marginal -> RE-QUERY"),
        ((0.028, -0.030), 1, 6.0, 1.0, "very noisy single -> low conf -> ABORT"),
        ((0.150,  0.020), 3, 1.2, 0.6, "gross offset -> NO_OBJECT -> widen search"),
    ]
    for i, (sab, k, ps, ac, label) in enumerate(cases, 1):
        robot, _ = build_world(BASE_XY, 0.0)
        r = run_episode(robot, BASE_XY, 0.0, sab, vlm, closed=True, temporal_frames=k,
                        use_gate=True, use_routing=True, perc_pos_cm=ps, act_pos_cm=ac,
                        mcs=mcs, tag=f"q{i}")
        print(f"  case {i}: {label:45s} -> success={r['success']} recov={r['recoveries']}")

    print("\n-- sim-to-real noise ablation --")
    mults, conds = [0.0, 1.0, 2.0, 3.0], ["open", "single", "temporal"]
    results = {c: [] for c in conds}
    for m in mults:
        ps, ac = ABL_PERC * m, ABL_ACT * m
        line = f"  noise x{m:.0f}: "
        for c in conds:
            rng = random.Random(1000); succ = 0
            for _ in range(ablation_n):
                robot, _ = build_world(BASE_XY, 0.0)
                sab = (rng.uniform(-0.045, 0.045), rng.uniform(-0.045, 0.045))
                r = run_episode(robot, BASE_XY, 0.0, sab, vlm,
                                closed=(c != "open"),
                                temporal_frames=3 if c == "temporal" else 1,
                                use_gate=(c != "open"), perc_pos_cm=ps, act_pos_cm=ac,
                                tag="abl")
                succ += int(r["success"])
            rate = 100 * succ / ablation_n; results[c].append(rate)
            line += f"{c[:4]}={rate:3.0f}%  "
        print(line)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    sty = {"open": ("#c0563b", "o", "Open-loop"), "single": ("#d69b3b", "s", "Closed single-frame"),
           "temporal": ("#3b7dc0", "^", "Closed 3-frame temporal")}
    for c in conds:
        col, mk, lab = sty[c]
        ax.plot(mults, results[c], marker=mk, color=col, lw=2, ms=8, label=lab)
    ax.set_xlabel("noise multiplier"); ax.set_ylabel("success rate (%)")
    ax.set_title("Sim-to-real noise ablation"); ax.set_ylim(0, 100); ax.set_xticks(mults)
    ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim2real_ablation.png"), dpi=130); plt.close(fig)
    print(f"  wrote sim2real_ablation.png and mcs_faults.jsonl ({len(mcs.records)} records)")


# ==============================================================================
# SECTION 8 -- EXPERIMENT: TRIALS  (from demo_trials.py)
#   Highlights: N randomized trials at different LOCATIONS and ORIENTATIONS
#   (3-DOF, dx/dy/dtheta correction), reported with 95% Wilson confidence
#   intervals; writes a workspace map, a summary chart, and a CSV.
# ==============================================================================

def experiment_trials(n=120):
    print("=" * 70 + f"\nTRIALS (from demo_trials.py): {n} location+orientation trials\n" + "=" * 70)
    vlm = make_vlm(); rng = random.Random(42); rows = []
    for i in range(1, n + 1):
        ox, oy = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)
        oyaw = math.radians(rng.uniform(*YAW_RANGE_DEG))
        sab = (rng.uniform(-SAB_POS, SAB_POS), rng.uniform(-SAB_POS, SAB_POS),
               math.radians(rng.uniform(-SAB_YAW_DEG, SAB_YAW_DEG)))
        robot, _ = build_world((ox, oy), oyaw)
        ro = run_episode(robot, (ox, oy), oyaw, sab, vlm, closed=False, use_orientation=True,
                         perc_pos_cm=TRIALS_PERC_POS, perc_yaw_deg=TRIALS_PERC_YAW,
                         act_pos_cm=TRIALS_ACT_POS, act_yaw_deg=TRIALS_ACT_YAW, tag=f"o{i}")
        robot, _ = build_world((ox, oy), oyaw)
        rc = run_episode(robot, (ox, oy), oyaw, sab, vlm, closed=True, use_orientation=True,
                         temporal_frames=3, use_gate=True, use_routing=True,
                         perc_pos_cm=TRIALS_PERC_POS, perc_yaw_deg=TRIALS_PERC_YAW,
                         act_pos_cm=TRIALS_ACT_POS, act_yaw_deg=TRIALS_ACT_YAW, tag=f"c{i}")
        rows.append({"trial": i, "x": round(ox, 3), "y": round(oy, 3),
                     "yaw_deg": round(math.degrees(oyaw), 1),
                     "open_success": ro["success"], "closed_success": rc["success"],
                     "recoveries": rc["recoveries"], "failure_type": rc["ftype"]})

    o = sum(r["open_success"] for r in rows); c = sum(r["closed_success"] for r in rows)
    po, olo, ohi = wilson(o, n); pc, clo, chi = wilson(c, n)
    print(f"  Open-loop  : {o}/{n} = {100*po:.1f}%  (95% CI {100*olo:.1f}-{100*ohi:.1f}%)")
    print(f"  Closed-loop: {c}/{n} = {100*pc:.1f}%  (95% CI {100*clo:.1f}-{100*chi:.1f}%)")

    with open(os.path.join(OUT, "trials_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _bar_chart(os.path.join(OUT, "trials_summary.png"), ["Open-loop", "Closed-loop"],
               [100 * po, 100 * pc], cis=[(100 * olo, 100 * ohi), (100 * clo, 100 * chi)],
               title=f"{n} location+orientation trials (95% CI)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for r in rows:
        yaw = math.radians(r["yaw_deg"])
        ax.quiver(r["x"], r["y"], math.cos(yaw), math.sin(yaw),
                  color="#3b7dc0" if r["closed_success"] else "#c0563b",
                  angles="xy", scale=22, width=0.006)
        if not r["open_success"]:
            ax.scatter(r["x"], r["y"], s=130, facecolors="none", edgecolors="#999", lw=0.7, zorder=0)
    ax.set_xlim(X_RANGE[0] - 0.03, X_RANGE[1] + 0.03); ax.set_ylim(Y_RANGE[0] - 0.05, Y_RANGE[1] + 0.05)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"{n} trials: location + orientation (arrow=yaw)\nblue=recovered, red=failed")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "workspace_map.png"), dpi=130); plt.close(fig)
    print("  wrote trials_results.csv, trials_summary.png, workspace_map.png")


# ==============================================================================
# SECTION 9 -- TOOL: PROBE THE REAL VLM  (from probe_vlm.py)
#   Highlights: render one frame of a KNOWN offset, send to Llama, print the raw
#   JSON + latency. Use before full trials to verify Ollama and calibrate signs.
# ==============================================================================

def probe_real_vlm(image_path=None):
    if not _OLLAMA:
        print("ollama not installed; run: pip install ollama"); return
    if image_path is None:
        robot, _ = build_world((0.54, 0.03), 0.0)         # cube down-and-right of gripper
        move(robot, [0.50, 0.0, 0.22], 0.0)
        image_path = capture_rgbd(robot, os.path.join(OUT, "probe_frame.png"))
        print(f"rendered probe frame -> {image_path} (true offset ~ +4cm x, +3cm y)")
    t0 = time.perf_counter()
    try:
        raw = ollama.chat(model=VLM_MODEL, format="json",
                          messages=[{"role": "system", "content": PROMPTS[F_POS]},
                                    {"role": "user", "content": "Return the JSON.",
                                     "images": [image_path]}],
                          options={"temperature": 0.1})["message"]["content"]
    except Exception as e:
        print(f"model call failed: {e}\nDid you run:  ollama pull {VLM_MODEL} ?")
        return
    dt = time.perf_counter() - t0
    print("raw:", raw, f"\nlatency: {dt:.1f}s")
    try:
        d = _extract_json(raw)
        if d.get("reasoning"):
            print("VLM reasoning:", d.get("reasoning"))
        print(f"parsed -> dx_cm={d.get('dx_cm')}  dy_cm={d.get('dy_cm')}  "
              f"dtheta_deg={d.get('dtheta_deg')}  confidence={d.get('confidence')}")
        print("calibration check: the cube in probe_frame.png sits DOWN-AND-RIGHT of "
              "centre, so a correct reply has dx_cm > 0 and dy_cm < 0. If a sign is "
              "inverted, flip CAM_DX_SIGN / CAM_DY_SIGN in Section 1.")
    except Exception as e:
        print("could not parse JSON from reply:", e)


# ==============================================================================
# SECTION 10 -- VIDEO RENDERING  (from render_video.py / render_scenarios.py /
#   render_advanced_trace.py / render_trials_montage.py)
#   Highlights: a fixed scene camera films the recovery at several locations &
#   orientations, captions each phase, and encodes an MP4. (A held cube tracks
#   the gripper on success -- a display convenience.)
# ==============================================================================

def render_video(out_mp4=None):
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    out_mp4 = out_mp4 or os.path.join(OUT, "combined_montage.mp4")
    W, H, FPS = 720, 500, 30
    frames, cap = [], {"title": "", "state": ""}
    held = {"on": False, "obj": None, "yaw": 0.0}
    VIEW = p.computeViewMatrixFromYawPitchRoll([0.5, 0.0, 0.08], 1.3, 50, -35, 0, 2)
    PROJ = p.computeProjectionMatrixFOV(55, W / H, 0.1, 3.0)
    try:
        FT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
        FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        FT = FS = ImageFont.load_default()

    def grab(color=(150, 200, 255)):
        _, _, rgba, _, _ = p.getCameraImage(W, H, VIEW, PROJ, renderer=p.ER_TINY_RENDERER)
        img = Image.fromarray(np.reshape(rgba, (H, W, 4))[:, :, :3].astype("uint8"))
        d = ImageDraw.Draw(img); d.rectangle([0, 0, W, 52], fill=(16, 20, 28))
        d.text((12, 6), cap["title"], font=FT, fill=(255, 255, 255))
        d.text((12, 30), cap["state"], font=FS, fill=color)
        frames.append(np.asarray(img))

    def hstep(robot, n=1):
        for _ in range(n):
            p.stepSimulation()
            if held["on"]:
                e, _ = ee_pose(robot)
                p.resetBasePositionAndOrientation(held["obj"], [e[0], e[1], e[2] - 0.045],
                                                  p.getQuaternionFromEuler([0, 0, held["yaw"]]))

    def vmove(robot, xyz, yaw, steps=50, col=(150, 200, 255)):
        jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), yaw_to_quat(yaw))
        for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
        for i in range(steps):
            hstep(robot)
            if i % 4 == 0:
                grab(col)

    def vhold(robot, steps, col=(150, 200, 255)):
        for i in range(steps):
            hstep(robot)
            if i % 3 == 0:
                grab(col)

    def vgrasp(robot, obj, axy, ayaw, oxy, oyaw, col):
        held["on"] = False
        vmove(robot, [axy[0], axy[1], 0.25], ayaw, col=col)
        vmove(robot, [axy[0], axy[1], 0.055], ayaw, col=col)
        ok = (np.linalg.norm(np.array(axy) - np.array(oxy)) < CAPTURE_RADIUS
              and abs(math.degrees(ayaw - oyaw)) < YAW_TOL_DEG)
        for j in PANDA_FINGER_JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.015 if ok else 0.04, force=40)
        for i in range(15):
            hstep(robot)
            (i % 3 == 0) and grab(col)
        if ok:
            held.update(on=True, obj=obj, yaw=ayaw)
        vmove(robot, [axy[0], axy[1], 0.27], ayaw, col=col)
        vhold(robot, 10, col)
        return ok

    trials = [((0.44, 0.16), 30, (0.035, 0.02, math.radians(18)), "back-left"),
              ((0.57, -0.15), -25, (-0.03, -0.025, math.radians(-20)), "front-right"),
              ((0.50, 0.0), 40, (0.03, 0.03, math.radians(22)), "centre")]
    recovered = 0
    for i, (oxy, oyaw_d, sab, corner) in enumerate(trials, 1):
        oyaw = math.radians(oyaw_d); robot, obj = build_world(oxy, oyaw)
        cap["title"] = f"Trial {i}/{len(trials)} [{corner}]  obj ({oxy[0]:.2f},{oxy[1]:+.2f}) yaw {oyaw_d:+d}\u00b0"
        aim = (oxy[0] + sab[0], oxy[1] + sab[1]); ayaw = oyaw + sab[2]
        cap["state"] = "saboteur miss (position + orientation)"
        vgrasp(robot, obj, aim, ayaw, oxy, oyaw, (240, 205, 100)); vhold(robot, 14, (240, 205, 100))
        dx = (oxy[0] - aim[0]) * 100 + np.random.normal(0, 0.5)
        dy = (oxy[1] - aim[1]) * 100 + np.random.normal(0, 0.5)
        dth = math.degrees(oyaw - ayaw) + np.random.normal(0, 1.5)
        cap["state"] = f"VLM correction dx={dx:+.1f} dy={dy:+.1f}cm d\u03b8={dth:+.0f}\u00b0 -> retry"
        vhold(robot, 14)
        ok = vgrasp(robot, obj, (aim[0] + dx / 100, aim[1] + dy / 100), ayaw + math.radians(dth),
                    oxy, oyaw, (110, 220, 130))
        recovered += int(ok); cap["state"] = f"RECOVERED [{recovered}/{i}]"
        vhold(robot, 24, (110, 220, 130))
    imageio.mimsave(out_mp4, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"  wrote {out_mp4} ({len(frames)} frames, recovered {recovered}/{len(trials)})")


def render_camera_view(out_mp4=None):
    """Eye-in-hand camera view: films WHAT THE VLM SEES during recovery. The left
    panel is the gripper camera with a crosshair at the image centre (= the gripper
    target); the object sitting off-crosshair IS the error, and after the correction
    it sits on the crosshair. The right panels give scene context + the diagnosis."""
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    out_mp4 = out_mp4 or os.path.join(OUT, "camera_view_recovery.mp4")
    W, H, FPS, CAM = 980, 570, 30, 470
    SCN_W, SCN_H = 400, 300
    frames = []
    ui = {"title": "", "state": "", "diag": "", "col": (150, 200, 255)}
    held = {"on": False, "obj": None, "yaw": 0.0}
    SCENE_VIEW = p.computeViewMatrixFromYawPitchRoll([0.50, 0.0, 0.08], 1.25, 50, -35, 0, 2)
    SCENE_PROJ = p.computeProjectionMatrixFOV(55, SCN_W / SCN_H, 0.1, 3.0)

    def _f(sz, b=False):
        base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        try:
            return ImageFont.truetype(base + ("-Bold.ttf" if b else ".ttf"), sz)
        except Exception:
            return ImageFont.load_default()
    FT, FL, FB, FM = _f(20, True), _f(12, True), _f(14), _f(12)

    def eye_in_hand(robot):
        pos, orn = ee_pose(robot)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        fwd, up = rot @ np.array([0, 0, 1]), rot @ np.array([0, 1, 0])
        eye = pos + fwd * 0.05
        view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
        _, _, rgba, _, _ = p.getCameraImage(CAM, CAM, view, proj, renderer=p.ER_TINY_RENDERER)
        return Image.fromarray(np.reshape(rgba, (CAM, CAM, 4))[:, :, :3].astype("uint8"))

    def scene_shot():
        _, _, rgba, _, _ = p.getCameraImage(SCN_W, SCN_H, SCENE_VIEW, SCENE_PROJ,
                                            renderer=p.ER_TINY_RENDERER)
        return Image.fromarray(np.reshape(rgba, (SCN_H, SCN_W, 4))[:, :, :3].astype("uint8"))

    def grab(robot):
        canvas = Image.new("RGB", (W, H), (17, 21, 29)); d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, W, 50], fill=(11, 14, 20))
        d.text((18, 6), ui["title"], font=FT, fill=(255, 255, 255))
        d.text((18, 30), ui["state"], font=FM, fill=ui["col"])
        cx0, cy0 = 18, 70
        canvas.paste(eye_in_hand(robot), (cx0, cy0))
        d.rectangle([cx0 - 1, cy0 - 1, cx0 + CAM, cy0 + CAM], outline=(60, 74, 92), width=1)
        d.text((cx0, cy0 - 18), "EYE-IN-HAND CAMERA  \u2014  what the VLM sees",
               font=FL, fill=(230, 130, 60))
        mx, my = cx0 + CAM // 2, cy0 + CAM // 2; ch = (255, 90, 60)
        for dd in (-1, 0, 1):
            d.line([mx + dd, my - 26, mx + dd, my - 8], fill=ch)
            d.line([mx + dd, my + 8, mx + dd, my + 26], fill=ch)
            d.line([mx - 26, my + dd, mx - 8, my + dd], fill=ch)
            d.line([mx + 8, my + dd, mx + 26, my + dd], fill=ch)
        d.ellipse([mx - 5, my - 5, mx + 5, my + 5], outline=ch, width=2)
        d.text((mx + 30, my - 8), "gripper target", font=FM, fill=ch)
        sx0, sy0 = CAM + 44, 70
        canvas.paste(scene_shot(), (sx0, sy0))
        d.rectangle([sx0 - 1, sy0 - 1, sx0 + SCN_W, sy0 + SCN_H], outline=(60, 74, 92), width=1)
        d.text((sx0, sy0 - 18), "SCENE VIEW  \u2014  context", font=FL, fill=(120, 145, 175))
        py = sy0 + SCN_H + 22
        d.rectangle([sx0, py, sx0 + SCN_W, py + 148], fill=(11, 14, 20), outline=(45, 58, 74))
        d.text((sx0 + 14, py + 10), "VLM DIAGNOSIS", font=FL, fill=(120, 145, 175))
        yy = py + 34
        for line in ui["diag"].split("\n"):
            d.text((sx0 + 14, yy), line, font=FB, fill=(215, 228, 242)); yy += 22
        frames.append(np.asarray(canvas))

    def hstep(robot, n=1):
        for _ in range(n):
            p.stepSimulation()
            if held["on"]:
                e, _ = ee_pose(robot)
                p.resetBasePositionAndOrientation(held["obj"], [e[0], e[1], e[2] - 0.045],
                                                  p.getQuaternionFromEuler([0, 0, held["yaw"]]))

    def vmove(robot, xyz, yaw, steps=48, cap=6):
        jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), yaw_to_quat(yaw))
        for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
        for i in range(steps):
            hstep(robot); (i % cap == 0) and grab(robot)

    def vhold(robot, steps):
        for i in range(steps):
            hstep(robot); (i % 3 == 0) and grab(robot)

    def try_grasp(robot, obj, axy, ayaw, oxy, oyaw):
        held["on"] = False
        vmove(robot, [axy[0], axy[1], 0.25], ayaw); vmove(robot, [axy[0], axy[1], 0.055], ayaw)
        ok = (np.linalg.norm(np.array(axy) - np.array(oxy)) < CAPTURE_RADIUS
              and abs(math.degrees(ayaw - oyaw)) < YAW_TOL_DEG)
        for j in PANDA_FINGER_JOINTS:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.015 if ok else 0.04, force=40)
        for i in range(14):
            hstep(robot); (i % 3 == 0) and grab(robot)
        if ok:
            held.update(on=True, obj=obj, yaw=ayaw)
        vmove(robot, [axy[0], axy[1], 0.27], ayaw); vhold(robot, 8)
        return ok

    scenarios = [((0.50, 0.00), 0, (0.040, -0.018, 14), "centre \u00b7 offset RIGHT"),
                 ((0.45, 0.15), 28, (-0.020, 0.038, -18), "back-left \u00b7 offset FORWARD"),
                 ((0.56, -0.13), -22, (0.034, 0.026, 20), "front-right \u00b7 offset DIAGONAL")]
    AMBER, BLUE, GREEN = (240, 205, 100), (150, 200, 255), (110, 220, 130)
    recovered = 0
    for i, (oxy, oyaw_d, sab, label) in enumerate(scenarios, 1):
        oyaw = math.radians(oyaw_d); robot, obj = build_world(oxy, oyaw)
        ui["title"] = f"Scenario {i}/{len(scenarios)}   \u2014   {label}"
        aim = (oxy[0] + sab[0], oxy[1] + sab[1]); ayaw = oyaw + math.radians(sab[2])
        ui["col"] = AMBER
        ui["state"] = f"FSM: APPROACH \u2192 GRASP  |  object ({oxy[0]:.2f},{oxy[1]:+.2f})m yaw {oyaw_d:+d}\u00b0"
        ui["diag"] = (f"saboteur injected:\n  {math.hypot(sab[0], sab[1])*100:.1f} cm, {sab[2]:+d}\u00b0 yaw\n"
                      "watch: object sits AWAY\nfrom the crosshair \u2192 miss")
        try_grasp(robot, obj, aim, ayaw, oxy, oyaw)
        ui["state"] = "FSM: CHECK \u2192 MISS. Lifting to diagnostic hover..."
        ui["diag"] = "grasp FAILED\n\nopen-loop would stop here\n(binary fault code)"
        vmove(robot, [aim[0], aim[1], 0.22], ayaw); vhold(robot, 12)
        ui["col"] = BLUE
        ui["state"] = "FSM: DIAGNOSE  |  sending frames to the VLM"
        ui["diag"] = "capturing eye-in-hand frames...\n\nthe object visible off-crosshair\nIS the error being measured"
        vhold(robot, 18)
        dx = (oxy[0] - aim[0]) * 100; dy = (oxy[1] - aim[1]) * 100; dth = -sab[2]
        conf = 0.75 + 0.2 * np.random.rand()
        ui["state"] = f"CONFIDENCE GATE: {conf:.2f} \u2265 0.60  \u2192  ACT"
        ui["diag"] = f"dx = {dx:+.1f} cm\ndy = {dy:+.1f} cm\nd\u03b8 = {dth:+.0f}\u00b0\nconfidence = {conf:.2f}"
        vhold(robot, 20)
        ui["col"] = GREEN
        ui["state"] = "FSM: RECOVER  |  applying correction and retrying"
        ok = try_grasp(robot, obj, (aim[0] + dx / 100, aim[1] + dy / 100),
                       ayaw + math.radians(dth), oxy, oyaw)
        recovered += int(ok)
        ui["state"] = f"RECOVERED  \u2014  object now centred on the crosshair  [{recovered}/{i}]"
        ui["diag"] = "correction applied\n\nobject is now ON the crosshair\n\u2192 grasped and lifted"
        vhold(robot, 22)
    imageio.mimsave(out_mp4, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"  wrote {out_mp4} ({len(frames)} frames, recovered {recovered}/{len(scenarios)})")


# ==============================================================================
# SECTION 11 -- DISPATCHER  (choose which experiment to run)
# ==============================================================================

def main():
    global _DOWN
    ap = argparse.ArgumentParser(description="Autonomous failure recovery -- all-in-one")
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["base", "advanced", "trials", "probe", "video", "all"])
    ap.add_argument("--n", type=int, default=120, help="trials count (trials mode)")
    ap.add_argument("--image", default=None, help="frame path (probe mode)")
    ap.add_argument("--view", default="scene", choices=["scene", "camera", "both"],
                    help="video mode: scene montage, eye-in-hand camera view, or both")
    args = ap.parse_args()

    p.connect(p.DIRECT)
    _DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
    np.random.seed(0); random.seed(0)
    print(f"VLM backend: {'RealVLM (Llama 3.2 Vision)' if USE_REAL_VLM else 'MockVLM'}\n")

    if args.mode in ("base", "all"):
        experiment_base()
    if args.mode in ("advanced", "all"):
        experiment_advanced()
    if args.mode in ("trials", "all"):
        experiment_trials(args.n)
    if args.mode == "probe":
        probe_real_vlm(args.image)
    if args.mode == "video":
        if args.view in ("scene", "both"):
            render_video()
        if args.view in ("camera", "both"):
            render_camera_view()

    p.disconnect()


if __name__ == "__main__":
    main()
