"""
DEMO: Autonomous Failure Recovery in Robotic Manipulation (PyBullet alternative)
================================================================================
Runs headless. Everything is REAL PyBullet. The VLM has two interchangeable
backends selected by ONE flag, USE_REAL_VLM (see the VLM SELECTION block):
  * MockVLM  -- ground truth + noise, works anywhere (default).
  * RealVLM  -- real Llama 3.2 Vision via Ollama; flip USE_REAL_VLM=True to run
               against the actual model with no other code change.

What is REAL here:
  - PyBullet physics, Franka Panda load, inverse kinematics
  - Eye-in-hand RGB-D camera rendering (a real PNG is saved)
  - The saboteur (deliberate target offset -> missed grasps)
  - The finite-state closed-loop recovery machine
  - The open-loop vs closed-loop evaluation harness

What is MOCKED (cannot run without Ollama + GPU + 7.8GB model):
  - The VLM's visual diagnosis. MockVLM returns the same
    {"reasoning","dx_cm","dy_cm","confidence"} dict the real model would,
    computed from ground truth + noise to emulate an imperfect estimate.

Grasp success model: a top-down "capture" model -- the grasp succeeds if the
gripper's final XY lands within CAPTURE_RADIUS of the object. This keeps the
demo deterministic and avoids fighting contact-dynamics tuning; the handbook's
grasp_succeeded() uses lift-height with full contact physics instead.
"""

import os
import time
import random
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

random.seed(7)
np.random.seed(7)

OUT = Path(__file__).resolve().parent / "outputs"
FRAMES_DIR = OUT / "frames"
OUT.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# ---- Panda constants (bundled panda.urdf) ----
PANDA_EE_LINK = 11
PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]
FINGER_OPEN = 0.04
DOWN_ORN = None  # set after connect

IMG_W, IMG_H = 320, 320
CAM_OFFSET = 0.05
CAPTURE_RADIUS = 0.028   # metres: gripper within this of object == caught
MAX_RECOVERIES = 3
BASE_XY = (0.5, 0.0)

# ============================================================================
#  VLM SELECTION  --  flip this ONE flag to run against the real model.
# ============================================================================
#  USE_REAL_VLM = False  -> MockVLM (works anywhere, no GPU/Ollama needed)
#  USE_REAL_VLM = True   -> RealVLM -> Llama 3.2 Vision via Ollama
#
#  To go live on your Windows machine:
#    1) install Ollama for Windows, then:  ollama pull llama3.2-vision
#    2) pip install ollama
#    3) set USE_REAL_VLM = True below and run  python demo.py
#  Nothing else in the script changes.
# ----------------------------------------------------------------------------
USE_REAL_VLM = False
VLM_MODEL = "llama3.2-vision"
DIAGNOSE_HOVER_Z = 0.22      # height to lift to before the VLM "looks" at the miss

# The real model returns a correction in CAMERA axes (+x = image right,
# +y = image up). For our top-down eye-in-hand camera these map onto the base
# axes up to a sign. Calibrate these ONCE with the real model: if a correction
# makes the miss WORSE instead of better, flip the offending sign.
CAM_DX_SIGN = +1.0
CAM_DY_SIGN = +1.0

# ollama is optional: guard the import so the mock path runs even without it.
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# Constrained chain-of-thought prompt (identical to vlm/diagnose.py in the handbook).
SYSTEM_PROMPT = """You are a robotics failure-diagnosis module for a top-down
pick-and-place robot with a downward-facing eye-in-hand camera.

The gripper just attempted a grasp and MISSED the object. In the image, the
object is visible but is NOT centred under the gripper (the image centre is the
gripper's current target).

Reason step by step about which direction the gripper must move to centre on the
object, THEN output a correction. The camera frame axes are:
  +x = image right,  +y = image up.
Give the correction as the offset, in centimetres, to ADD to the current target
so the gripper lands on the object.

Respond with ONLY a JSON object of this exact shape:
{"reasoning": "<one short sentence>", "dx_cm": <number>, "dy_cm": <number>, "confidence": <0-1>}
"""


# ---------------------------------------------------------------- world / camera
def build_world():
    """Create a fresh PyBullet world with the floor and Franka Panda robot."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    rest = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
    for j, a in zip(PANDA_ARM_JOINTS, rest):
        p.resetJointState(robot, j, a)
    for j in PANDA_FINGER_JOINTS:
        p.resetJointState(robot, j, FINGER_OPEN)
    return robot


def spawn_object(xy):
    """Place the target cube at the requested XY location."""
    obj = p.loadURDF("cube_small.urdf", basePosition=[xy[0], xy[1], 0.02])
    # High-contrast colour so the real VLM can pick the object out of the frame.
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    return obj


def get_ee_pose(robot):
    """Return the end-effector position and orientation."""
    s = p.getLinkState(robot, PANDA_EE_LINK, computeForwardKinematics=True)
    return np.array(s[4]), s[5]


def capture_rgbd(robot, save_path=None):
    """Render the gripper-mounted RGB-D camera and optionally save the RGB image."""
    ee_pos, ee_orn = get_ee_pose(robot)
    rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    forward = rot @ np.array([0, 0, 1])
    up = rot @ np.array([0, 1, 0])
    eye = ee_pos + forward * CAM_OFFSET
    target = eye + forward * 0.5
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, IMG_W / IMG_H, 0.02, 2.0)
    w, h, rgba, depth_buf, _ = p.getCameraImage(
        IMG_W, IMG_H, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
    near, far = 0.02, 2.0
    z = np.reshape(depth_buf, (h, w))
    depth = far * near / (far - (far - near) * z)
    if save_path:
        Image.fromarray(rgb).save(save_path)
    return rgb, depth


# ---------------------------------------------------------------- grasp primitive
def _move_to(robot, xyz, steps=120):
    """Move the gripper toward a Cartesian target using inverse kinematics."""
    jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), DOWN_ORN)
    for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for _ in range(steps):
        p.stepSimulation()


def attempt_grasp(robot, grasp_xy, sabotage=None):
    """Run one top-down grasp, optionally offset by a saboteur miss."""
    gx, gy = grasp_xy
    if sabotage is not None:
        gx += sabotage[0]
        gy += sabotage[1]
    _move_to(robot, [gx, gy, 0.25])      # hover
    _move_to(robot, [gx, gy, 0.05])      # descend
    ee_pos, _ = get_ee_pose(robot)       # where the gripper actually ended up
    return np.array([gx, gy]), ee_pos[:2]


def grasp_succeeded(gripper_xy, object_xy):
    """Check whether the landing point is close enough to count as a grasp."""
    return float(np.linalg.norm(np.array(gripper_xy) - np.array(object_xy))) < CAPTURE_RADIUS


# ============================================================================
#  VLM BACKENDS  --  both expose the SAME method:
#      diagnose(image_path, gripper_xy=None, object_xy=None) -> dict
#  RealVLM uses ONLY the image (like a real deployment).
#  MockVLM uses ground truth to emulate the model without needing a GPU.
# ============================================================================
import json


class RealVLM:
    """Real diagnosis via Llama 3.2 Vision through Ollama. This is the block that
    runs against the actual model -- no code changes needed, just USE_REAL_VLM=True
    and a running Ollama with the model pulled."""
    def __init__(self, model=VLM_MODEL):
        """Verify the Ollama Python package is available and store the model name."""
        if not _OLLAMA_AVAILABLE:
            raise RuntimeError(
                "USE_REAL_VLM=True but the 'ollama' package is not installed. "
                "Run: pip install ollama   (and install Ollama for Windows, then "
                "`ollama pull llama3.2-vision`).")
        self.model = model
        self.calls = 0

    def diagnose(self, image_path, gripper_xy=None, object_xy=None):
        """image_path -> {reasoning, dx_cm, dy_cm, confidence, _latency_s} or None."""
        self.calls += 1
        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model=self.model,
                format="json",                      # forces valid JSON output
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": "Diagnose the missed grasp and return the correction JSON.",
                     "images": [image_path]},
                ],
                options={"temperature": 0.1},
            )
            data = json.loads(resp["message"]["content"])
            data["dx_cm"] = float(data["dx_cm"])
            data["dy_cm"] = float(data["dy_cm"])
            data["_latency_s"] = time.perf_counter() - t0
            return data
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"    [RealVLM] unparseable/failed response: {e}")
            return None


class MockVLM:
    """Emulates Llama 3.2 Vision from ground truth + noise (no GPU/Ollama needed).
    Same return shape as RealVLM so the two are drop-in interchangeable."""
    def __init__(self, noise_cm=2.0):   # ~2cm std: emulates an imperfect estimate
        """Configure the standard deviation of mock correction noise."""
        self.noise_cm = noise_cm
        self.calls = 0

    def diagnose(self, image_path=None, gripper_xy=None, object_xy=None):
        """Return a noisy correction from the gripper target to the object."""
        self.calls += 1
        t0 = time.perf_counter()
        true = (np.array(object_xy) - np.array(gripper_xy)) * 100.0   # cm
        est = true + np.random.normal(0, self.noise_cm, size=2)
        return {
            "reasoning": "object appears offset from gripper centre; shifting to recentre",
            "dx_cm": float(est[0]),
            "dy_cm": float(est[1]),
            "confidence": 0.8,
            "_latency_s": time.perf_counter() - t0,
        }


def make_vlm():
    """One switch decides everything: MockVLM vs the real Llama 3.2 Vision."""
    return RealVLM() if USE_REAL_VLM else MockVLM()


def camera_to_base(dx_cm, dy_cm):
    """Top-down eye-in-hand -> camera axes align with base axes, so the planar
    correction maps through directly (the handbook uses the full 4x4 T_base_cam).
    CAM_D*_SIGN let you flip a sign once when calibrating the real model."""
    return np.array([CAM_DX_SIGN * dx_cm, CAM_DY_SIGN * dy_cm]) / 100.0


# ---------------------------------------------------------------- FSM episode
def run_episode(robot, object_xy, grasp_xy, sabotage, closed_loop, vlm, frame_tag=""):
    """Run one grasp attempt, optionally followed by VLM closed-loop recovery."""
    target, gripper_xy = attempt_grasp(robot, grasp_xy, sabotage=sabotage)
    if grasp_succeeded(gripper_xy, object_xy):
        return {"success": True, "attempts": 1, "recoveries": 0, "vlm_time": 0.0}
    if not closed_loop:
        return {"success": False, "attempts": 1, "recoveries": 0, "vlm_time": 0.0}

    total_vlm = 0.0
    for k in range(1, MAX_RECOVERIES + 1):
        # Lift to a hover so the eye-in-hand camera sees the whole workspace,
        # then render the real diagnostic frame the VLM will look at.
        _move_to(robot, [target[0], target[1], DIAGNOSE_HOVER_Z])
        frame = FRAMES_DIR / f"recover_{frame_tag}_{k}.png"
        capture_rgbd(robot, save_path=frame)                 # real render

        # Same call for mock OR real model. RealVLM ignores the ground-truth args.
        diag = vlm.diagnose(frame, gripper_xy, object_xy)
        if diag is None:                                     # model failed/garbled
            continue                                         # try another frame
        total_vlm += diag["_latency_s"]

        corr = camera_to_base(diag["dx_cm"], diag["dy_cm"])
        target = target + corr
        _, gripper_xy = attempt_grasp(robot, tuple(target))  # retry, no sabotage
        if grasp_succeeded(gripper_xy, object_xy):
            return {"success": True, "attempts": 1 + k, "recoveries": k,
                    "vlm_time": total_vlm}
    return {"success": False, "attempts": 1 + MAX_RECOVERIES,
            "recoveries": MAX_RECOVERIES, "vlm_time": total_vlm}


# ---------------------------------------------------------------- driver
def rand_sabotage():
    """Generate a random XY offset for the first grasp attempt."""
    return (random.uniform(-0.032, 0.032), random.uniform(-0.032, 0.032))


def main():
    """Run the baseline demo, save example images/plots, and print results."""
    global DOWN_ORN
    p.connect(p.DIRECT)
    DOWN_ORN = p.getQuaternionFromEuler([np.pi, 0, 0])
    vlm = make_vlm()
    backend = "REAL Llama 3.2 Vision (Ollama)" if USE_REAL_VLM else "MockVLM (ground-truth + noise)"
    print(f"VLM backend: {backend}\n")

    print("=" * 70)
    print("PART 1  Sanity: camera + clean grasp + one sabotaged grasp")
    print("=" * 70)

    robot = build_world()
    spawn_object(BASE_XY)
    for _ in range(120):
        p.stepSimulation()
    # move over the cube and save a real eye-in-hand frame
    _move_to(robot, [BASE_XY[0], BASE_XY[1], 0.25])
    capture_rgbd(robot, save_path=OUT / "eye_in_hand_view.png")
    print(f"  saved real eye-in-hand RGB frame -> eye_in_hand_view.png")

    robot = build_world(); spawn_object(BASE_XY)
    _, g = attempt_grasp(robot, BASE_XY)
    print(f"  clean grasp        -> success={grasp_succeeded(g, BASE_XY)}")

    robot = build_world(); spawn_object(BASE_XY)
    sab = (0.04, 0.0)
    _, g = attempt_grasp(robot, BASE_XY, sabotage=sab)
    print(f"  sabotaged grasp {sab} -> success={grasp_succeeded(g, BASE_XY)} (should miss)")

    print()
    print("=" * 70)
    print("PART 2  Single closed-loop recovery, step by step")
    print("=" * 70)
    robot = build_world(); spawn_object(BASE_XY)
    sab = (0.038, -0.022)   # magnitude ~4.4cm > capture radius -> guaranteed miss
    print(f"  saboteur injects offset dx={sab[0]*100:+.1f}cm dy={sab[1]*100:+.1f}cm "
          f"(|offset|={np.hypot(*sab)*100:.1f}cm > capture {CAPTURE_RADIUS*100:.1f}cm -> will miss)")
    res = run_episode(robot, BASE_XY, BASE_XY, sab, closed_loop=True, vlm=vlm, frame_tag="single")
    print(f"  result: success={res['success']} after {res['recoveries']} recovery attempt(s), "
          f"VLM latency {res['vlm_time']*1000:.1f} ms")

    print()
    print("=" * 70)
    print("PART 3  Evaluation: open-loop baseline vs closed-loop recovery")
    print("=" * 70)
    N = 30
    summary = {}
    for cond, closed in [("open_loop", False), ("closed_loop", True)]:
        succ = 0; recov = []; vtimes = []
        # identical saboteur sequence for both conditions (fair comparison)
        rng = random.Random(42)
        for i in range(N):
            robot = build_world(); spawn_object(BASE_XY)
            sab = (rng.uniform(-0.032, 0.032), rng.uniform(-0.032, 0.032))
            r = run_episode(robot, BASE_XY, BASE_XY, sab, closed_loop=closed,
                            vlm=vlm, frame_tag=f"{cond}{i}")
            succ += int(r["success"])
            recov.append(r["recoveries"])
            if r["vlm_time"] > 0:
                vtimes.append(r["vlm_time"])
        rate = 100 * succ / N
        avg_lat = (sum(vtimes) / len(vtimes) * 1000) if vtimes else 0
        summary[cond] = (rate, succ, N, np.mean(recov), avg_lat)
        print(f"  {cond:12s}: {succ:2d}/{N} = {rate:4.0f}%   "
              f"mean recoveries {np.mean(recov):.2f}   "
              f"mean VLM latency {avg_lat:.1f} ms")

    ol = summary["open_loop"][0]; cl = summary["closed_loop"][0]
    print(f"\n  >> closed-loop recovered {cl-ol:.0f} percentage points of failures "
          f"({ol:.0f}% -> {cl:.0f}%)")

    p.disconnect()

    # --------- results chart ---------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    conds = ["Open-loop\n(baseline)", "Closed-loop\n(VLM recovery)"]
    rates = [summary["open_loop"][0], summary["closed_loop"][0]]
    bars = ax.bar(conds, rates, color=["#c0563b", "#3b7dc0"], width=0.55)
    ax.set_ylabel("Grasp success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Failure recovery: {N} sabotaged trials (identical offsets)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 2, f"{r:.0f}%",
                ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "results_open_vs_closed.png", dpi=130)
    print(f"\n  saved results chart -> results_open_vs_closed.png")


if __name__ == "__main__":
    main()
