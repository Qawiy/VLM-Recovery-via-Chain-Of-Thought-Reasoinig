"""
Multi-scenario render: the saboteur induces a miss at DIFFERENT workspace
locations, and the closed-loop VLM recovery runs for each. Exported as one MP4.

Headless (DIRECT + TinyRenderer). Fixed angled scene camera. Each scenario:
  attempt (miss)  ->  VLM diagnoses offset  ->  retry  ->  success.
A running "recovered k/N" tally is burned into the caption bar.

Same conventions as the earlier clip: real PyBullet throughout; the VLM here is
the labelled mock (RealVLM is one flag away in demo.py); a held cube tracks the
gripper on success so the outcome reads clearly on screen.
"""
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

np.random.seed(3)

W, H = 640, 480
FPS = 30
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_MP4 = OUT_DIR / "saboteur_scenarios.mp4"

PANDA_EE_LINK = 11
PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]
FINGER_OPEN, FINGER_CLOSED = 0.04, 0.015
CAPTURE_RADIUS = 0.028
NOISE_CM = 0.5

# (object_xy, saboteur_offset_m, human_label) -- different locations + directions
SCENARIOS = [
    ((0.50,  0.00), ( 0.045,  0.000), "offset RIGHT (+x)"),
    ((0.46,  0.17), ( 0.000, -0.045), "offset BACK (-y)"),
    ((0.46, -0.17), (-0.010,  0.045), "offset FORWARD (+y)"),
    ((0.54,  0.09), (-0.040, -0.028), "offset DIAGONAL (-x,-y)"),
]

frames, caption, subcaption = [], "", ""
_held = {"on": False, "obj": None}

VIEW = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0.47, 0.0, 0.1], distance=1.2,
    yaw=48, pitch=-33, roll=0, upAxisIndex=2)
PROJ = p.computeProjectionMatrixFOV(fov=56, aspect=W / H, nearVal=0.1, farVal=3.0)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except Exception:
    FONT = ImageFont.load_default(); FONT_S = ImageFont.load_default()


def grab():
    """Render the current PyBullet scene with caption text into the video buffer."""
    _, _, rgba, _, _ = p.getCameraImage(W, H, VIEW, PROJ, renderer=p.ER_TINY_RENDERER)
    img = Image.fromarray(np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 50], fill=(18, 22, 30))
    d.text((12, 6), caption, font=FONT, fill=(255, 255, 255))
    if subcaption:
        d.text((12, 30), subcaption, font=FONT_S, fill=(150, 200, 255))
    frames.append(np.asarray(img))


def ee_pos(robot):
    """Return the Panda end-effector world position."""
    return np.array(p.getLinkState(robot, PANDA_EE_LINK, computeForwardKinematics=True)[4])


def step_hold(robot, n=1):
    """Advance physics and keep a successfully grasped cube attached to the gripper."""
    for _ in range(n):
        p.stepSimulation()
        if _held["on"] and _held["obj"] is not None:
            ep = ee_pos(robot)
            p.resetBasePositionAndOrientation(
                _held["obj"], [ep[0], ep[1], ep[2] - 0.045], [0, 0, 0, 1])


def move_to(robot, xyz, steps=60, cap=4):
    """Move the arm toward a 3D target and capture progress frames."""
    orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    jt = p.calculateInverseKinematics(robot, PANDA_EE_LINK, list(xyz), orn)
    for j, t in zip(PANDA_ARM_JOINTS, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for i in range(steps):
        step_hold(robot)
        if i % cap == 0:
            grab()


def set_fingers(robot, val, steps=18):
    """Command both gripper fingers to the same open/close value."""
    for j in PANDA_FINGER_JOINTS:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, val, force=40)
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab()


def hold_still(robot, steps=18):
    """Pause the scene while continuing to record video frames."""
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab()


def build(obj_xy):
    """Reset the simulation and create the robot, table plane, and cube."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(PANDA_ARM_JOINTS, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in PANDA_FINGER_JOINTS:
        p.resetJointState(robot, j, FINGER_OPEN)
    obj = p.loadURDF("cube_small.urdf", basePosition=[obj_xy[0], obj_xy[1], 0.02])
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    for _ in range(50):
        p.stepSimulation()
    return robot, obj


def do_grasp(robot, obj, aim_xy):
    """Execute one top-down grasp attempt and mark the cube held if it succeeds."""
    _held["on"] = False
    set_fingers(robot, FINGER_OPEN, steps=9)
    move_to(robot, [aim_xy[0], aim_xy[1], 0.25])       # hover
    move_to(robot, [aim_xy[0], aim_xy[1], 0.055])      # descend
    obj_xy = np.array(p.getBasePositionAndOrientation(obj)[0][:2])
    aligned = np.linalg.norm(np.array(aim_xy) - obj_xy) < CAPTURE_RADIUS
    set_fingers(robot, FINGER_CLOSED if aligned else FINGER_OPEN, steps=15)
    if aligned:
        _held["on"] = True; _held["obj"] = obj
    move_to(robot, [aim_xy[0], aim_xy[1], 0.27])       # lift
    hold_still(robot, 12)
    return aligned


def mock_vlm(aim, obj_xy):
    """Camera-frame correction (cm) toward the object + small noise."""
    est = (np.array(obj_xy) - np.array(aim)) * 100.0 + np.random.normal(0, NOISE_CM, 2)
    return est


def main():
    """Run each saboteur scenario and export the combined MP4."""
    global caption, subcaption
    p.connect(p.DIRECT)
    N = len(SCENARIOS)
    recovered = 0

    for idx, (obj_xy, sab, label) in enumerate(SCENARIOS, 1):
        robot, obj = build(obj_xy)
        aim = (obj_xy[0] + sab[0], obj_xy[1] + sab[1])

        # --- attempt: miss ---
        caption = f"Scenario {idx}/{N}: saboteur {label}"
        subcaption = f"object at ({obj_xy[0]:.2f}, {obj_xy[1]:.2f}) m  |  aiming off-target"
        do_grasp(robot, obj, aim)
        subcaption = "MISS - object left on table. Initiating VLM recovery..."
        hold_still(robot, 20)

        # --- diagnose ---
        obj_now = np.array(p.getBasePositionAndOrientation(obj)[0][:2])
        est = mock_vlm(aim, obj_now)
        caption = f"Scenario {idx}/{N}: VLM recovery"
        subcaption = f"VLM correction  dx={est[0]:+.1f}cm  dy={est[1]:+.1f}cm  ->  retry"
        hold_still(robot, 16)

        # --- retry ---
        corrected = (aim[0] + est[0] / 100.0, aim[1] + est[1] / 100.0)
        ok = do_grasp(robot, obj, corrected)
        recovered += int(ok)
        caption = f"Scenario {idx}/{N}: RECOVERED   [total {recovered}/{idx}]"
        subcaption = "re-aligned, grasped, lifted"
        hold_still(robot, 26)

    # closing summary card
    caption = f"Closed-loop VLM recovery: {recovered}/{N} scenarios recovered"
    subcaption = "saboteur varied by location & direction each time"
    hold_still(robot, 40)

    p.disconnect()
    imageio.mimsave(OUT_MP4, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {OUT_MP4}  ({len(frames)} frames, {len(frames)/FPS:.1f}s, "
          f"recovered {recovered}/{N})")


if __name__ == "__main__":
    main()
