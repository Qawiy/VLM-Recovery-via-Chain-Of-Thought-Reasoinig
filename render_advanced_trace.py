"""
Narrated render of the ADVANCED qualitative trace. Four cases, each showing the
new machinery on screen: failure-type routing, temporal frame count, the
confidence gate's decision (ACT / RE-QUERY / ABORT), the NO_OBJECT widen-search,
and the semantic MCS messages as a scrolling overlay.

Headless (DIRECT + TinyRenderer), fixed scene camera. Real PyBullet; the VLM is
the labelled mock (RealVLM is one flag away). A held cube tracks the gripper on
success so outcomes read clearly; a miss genuinely isn't held.
"""
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

np.random.seed(0)

W, H = 780, 520
FPS = 30
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_MP4 = OUT_DIR / "advanced_trace.mp4"

EE = 11
ARM = [0, 1, 2, 3, 4, 5, 6]
FING = [9, 10]
F_OPEN, F_CLOSE = 0.04, 0.015
CAPTURE = 0.028
CONF_ACT, CONF_REQUERY = 0.60, 0.35
OUT_OF_FRAME_CM = 12.0
DOWN = None

frames = []
title = ""          # top line: which case
state = ""          # second line: current FSM state / gate decision
mcs_lines = []      # scrolling MCS log (bottom overlay)
_held = {"on": False, "obj": None}

VIEW = p.computeViewMatrixFromYawPitchRoll([0.47, 0.0, 0.1], 1.2, 48, -33, 0, 2)
PROJ = p.computeProjectionMatrixFOV(56, W / H, 0.1, 3.0)

def _font(sz, bold=False):
    """Load an overlay font, falling back to PIL's default if unavailable."""
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    try:
        return ImageFont.truetype(base + ("-Bold.ttf" if bold else ".ttf"), sz)
    except Exception:
        return ImageFont.load_default()

F_TITLE, F_STATE, F_MCS = _font(22, True), _font(18, True), _font(14)

GATE_COLOR = {"ACT": (90, 210, 120), "RE-QUERY": (240, 200, 90),
              "ABORT": (240, 110, 90), "": (150, 200, 255)}


def grab(gate=""):
    """Capture one simulation frame with the current FSM and MCS overlays."""
    _, _, rgba, _, _ = p.getCameraImage(W, H, VIEW, PROJ, renderer=p.ER_TINY_RENDERER)
    img = Image.fromarray(np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    # top banner
    d.rectangle([0, 0, W, 58], fill=(16, 20, 28))
    d.text((14, 6), title, font=F_TITLE, fill=(255, 255, 255))
    d.text((14, 33), state, font=F_STATE, fill=GATE_COLOR.get(gate, (150, 200, 255)))
    # bottom MCS log panel
    ph = 96
    d.rectangle([0, H - ph, W, H], fill=(12, 14, 20))
    d.text((14, H - ph + 6), "MCS semantic fault log", font=F_MCS, fill=(120, 140, 170))
    for i, ln in enumerate(mcs_lines[-4:]):
        d.text((14, H - ph + 26 + i * 17), ln, font=F_MCS, fill=(190, 205, 225))
    frames.append(np.asarray(img))


def ee(robot):
    """Return the Panda end-effector world position."""
    return np.array(p.getLinkState(robot, EE, computeForwardKinematics=True)[4])


def step_hold(robot, n=1):
    """Advance physics while keeping a grasped cube attached to the gripper."""
    for _ in range(n):
        p.stepSimulation()
        if _held["on"] and _held["obj"] is not None:
            e = ee(robot)
            p.resetBasePositionAndOrientation(_held["obj"], [e[0], e[1], e[2] - 0.045], [0, 0, 0, 1])


def move(robot, xyz, steps=54, cap=4, gate=""):
    """Move the robot arm toward a Cartesian target and record progress frames."""
    jt = p.calculateInverseKinematics(robot, EE, list(xyz), DOWN)
    for j, t in zip(ARM, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for i in range(steps):
        step_hold(robot)
        if i % cap == 0:
            grab(gate)


def fingers(robot, v, steps=15, gate=""):
    """Open or close the gripper fingers while recording frames."""
    for j in FING:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, v, force=40)
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab(gate)


def hold(robot, steps=18, gate=""):
    """Pause the animation while continuing to render frames."""
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab(gate)


def build(obj_xy):
    """Reset PyBullet and create the robot, floor, and cube for one case."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(); p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(ARM, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in FING:
        p.resetJointState(robot, j, F_OPEN)
    obj = p.loadURDF("cube_small.urdf", basePosition=[obj_xy[0], obj_xy[1], 0.02])
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    for _ in range(40):
        p.stepSimulation()
    return robot, obj


def mcs(msg):
    """Append a message to the on-screen semantic MCS fault log."""
    mcs_lines.append(msg)


def grasp(robot, obj, aim_xy, gate=""):
    """Attempt a top-down grasp and attach the cube if the aim is close enough."""
    _held["on"] = False
    fingers(robot, F_OPEN, 9, gate)
    move(robot, [aim_xy[0], aim_xy[1], 0.25], gate=gate)
    move(robot, [aim_xy[0], aim_xy[1], 0.055], gate=gate)
    obj_xy = np.array(p.getBasePositionAndOrientation(obj)[0][:2])
    aligned = np.linalg.norm(np.array(aim_xy) - obj_xy) < CAPTURE
    fingers(robot, F_CLOSE if aligned else F_OPEN, 15, gate)
    if aligned:
        _held["on"] = True; _held["obj"] = obj
    move(robot, [aim_xy[0], aim_xy[1], 0.27], gate=gate)
    hold(robot, 10, gate)
    return aligned


def classify(aim, obj_xy):
    """Classify the miss as positional or no-object based on offset size."""
    off = np.linalg.norm(np.array(obj_xy) - np.array(aim)) * 100
    return ("NO_OBJECT", 0.9) if off > OUT_OF_FRAME_CM else ("POSITIONAL_OFFSET", 0.85)


def diagnose(aim, obj_xy, k, perc_sigma):
    """Create a mock temporal VLM correction with confidence for the current miss."""
    true = (np.array(obj_xy) - np.array(aim)) * 100
    draws = [true + np.random.normal(0, perc_sigma, 2) for _ in range(k)]
    est = np.median(np.array(draws), axis=0)
    spread = float(np.mean(np.std(np.array(draws), axis=0))) if k > 1 else perc_sigma
    off = float(np.linalg.norm(true))
    conf = float(np.clip(0.95 - 0.09 * perc_sigma - 0.02 * off
                         + 0.07 * (k - 1) / (1 + spread) + np.random.normal(0, 0.02),
                         0.05, 0.99))
    return est, conf


def case(robot, obj, obj_xy, nominal, sab, k, perc_sigma, label):
    """Run one qualitative case with on-screen gating/routing/MCS. Returns outcome."""
    global title, state
    title = label
    aim = (obj_xy[0] + sab[0], obj_xy[1] + sab[1])

    state = f"FSM: APPROACH -> GRASP  (aiming off-target)"
    grasp(robot, obj, aim)
    state = "FSM: CHECK -> MISS.  Entering recovery."
    mcs("GRASP_FAILURE  open-loop would stop here (binary fault)")
    hold(robot, 16)

    target = np.array(aim, float)
    for attempt in range(1, 4):
        move(robot, [target[0], target[1], 0.22])           # hover to view
        ftype, fconf = classify(target, obj_xy)
        state = f"ROUTING: classified {ftype}  (conf {fconf:.2f})"
        hold(robot, 10)

        if ftype == "NO_OBJECT":
            mcs(f"FAILURE_ROUTED  NO_OBJECT -> widen search to nominal pose")
            state = "ROUTING: NO_OBJECT -> WIDEN SEARCH"
            hold(robot, 12)
            target = np.array(nominal, float)
            move(robot, [target[0], target[1], 0.22])
            ftype, fconf = classify(target, obj_xy)
            state = f"re-acquired: {ftype} (conf {fconf:.2f})"
            hold(robot, 10)

        est, conf = diagnose(target, obj_xy, k, perc_sigma)

        # confidence gate
        if conf < CONF_REQUERY:
            state = f"CONFIDENCE GATE: {conf:.2f} < {CONF_REQUERY:.2f}  ->  ABORT (unsafe)"
            mcs(f"RECOVERY_ABORTED  conf {conf:.2f} below safe threshold; no move made")
            hold(robot, 34, gate="ABORT")
            return "low_conf_abort"

        if conf < CONF_ACT:
            state = f"CONFIDENCE GATE: {conf:.2f} marginal  ->  RE-QUERY ({max(k,3)} frames)"
            mcs(f"DIAGNOSIS_REQUERIED  marginal conf {conf:.2f}; re-querying")
            hold(robot, 20, gate="RE-QUERY")
            est, conf = diagnose(target, obj_xy, max(k, 3), perc_sigma)
            state = f"re-query -> conf {conf:.2f}"
            hold(robot, 10, gate="RE-QUERY")
            if conf < CONF_REQUERY:
                mcs(f"RECOVERY_ABORTED  still low ({conf:.2f}); abort")
                hold(robot, 28, gate="ABORT")
                return "low_conf_abort"

        state = f"CONFIDENCE GATE: {conf:.2f} >= {CONF_ACT:.2f}  ->  ACT  (dx={est[0]:+.1f} dy={est[1]:+.1f}cm)"
        mcs(f"CORRECTION_APPLIED  {ftype} dx={est[0]:+.1f} dy={est[1]:+.1f}cm [{k}f] conf {conf:.2f}")
        hold(robot, 14, gate="ACT")
        target = target + est / 100.0
        ok = grasp(robot, obj, tuple(target), gate="ACT")
        if ok:
            state = f"RECOVERED on attempt {attempt}"
            mcs(f"RECOVERY_SUCCEEDED  resolved on attempt {attempt}")
            hold(robot, 30, gate="ACT")
            return "recovered"
    return "exhausted"


CASES = [
    ((0.50, 0.00), (0.50, 0.00), (0.038, -0.022), 3, 1.2,
     "Case 1: positional miss -> temporal (3f) diagnose -> ACT -> recover"),
    ((0.50, 0.00), (0.50, 0.00), (0.030,  0.030), 3, 3.6,
     "Case 2: high noise -> marginal confidence -> RE-QUERY -> recover"),
    ((0.50, 0.00), (0.50, 0.00), (0.028, -0.030), 1, 6.0,
     "Case 3: very noisy single-frame -> low confidence -> safe ABORT"),
    ((0.50, 0.00), (0.50, 0.00), (0.150,  0.020), 3, 1.2,
     "Case 4: gross offset -> NO_OBJECT routing -> widen search -> recover"),
]


def main():
    """Render all advanced qualitative cases into the MP4 output."""
    global DOWN, mcs_lines
    p.connect(p.DIRECT)
    DOWN = p.getQuaternionFromEuler([np.pi, 0, 0])
    for obj_xy, nom, sab, k, ps, label in CASES:
        robot, obj = build(obj_xy)
        outcome = case(robot, obj, obj_xy, nom, sab, k, ps, label)
        print(f"  {label[:40]:40s} -> {outcome}")
    p.disconnect()
    imageio.mimsave(OUT_MP4, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {OUT_MP4}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
