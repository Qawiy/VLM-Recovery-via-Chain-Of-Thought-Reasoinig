"""
Render the ROBOT'S CAMERA VIEW during VLM recovery, across several scenarios.

Layout per frame:
  LEFT  : the eye-in-hand RGB camera feed -- exactly what the VLM is sent.
          A crosshair marks the image centre = the gripper's current target.
          When the object sits away from the crosshair, that IS the failure the
          VLM diagnoses; after the correction it should sit ON the crosshair.
  RIGHT : a small scene view for context + the live FSM state / VLM correction.

Scenarios vary object LOCATION and ORIENTATION (yaw). Real PyBullet throughout;
the VLM is the labelled mock (RealVLM is one flag away in the experiment scripts).
"""
import math
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

np.random.seed(5)

# ---- canvas ----
W, H, FPS = 980, 570, 30
CAM = 470            # eye-in-hand panel (square)
SCN_W, SCN_H = 400, 300
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_MP4 = OUT_DIR / "camera_view_recovery.mp4"

EE = 11
ARM = [0, 1, 2, 3, 4, 5, 6]
FING = [9, 10]
F_OPEN, F_CLOSE = 0.04, 0.015
CAPTURE, YAW_TOL = 0.028, 12.0
HOVER_Z = 0.22

frames = []
ui = {"title": "", "state": "", "diag": "", "col": (150, 200, 255)}
held = {"on": False, "obj": None, "yaw": 0.0}

SCENE_VIEW = p.computeViewMatrixFromYawPitchRoll([0.50, 0.0, 0.08], 1.25, 50, -35, 0, 2)
SCENE_PROJ = p.computeProjectionMatrixFOV(55, SCN_W / SCN_H, 0.1, 3.0)


def _f(sz, b=False):
    """Load an overlay font, falling back to PIL's default if unavailable."""
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    try:
        return ImageFont.truetype(base + ("-Bold.ttf" if b else ".ttf"), sz)
    except Exception:
        return ImageFont.load_default()


FT, FL, FB, FM = _f(20, True), _f(12, True), _f(14), _f(12)


def yq(yaw):
    """Convert a planar yaw angle to the gripper-down quaternion."""
    return p.getQuaternionFromEuler([math.pi, 0, yaw])


def ee_pose(robot):
    """Return the Panda end-effector position and orientation."""
    s = p.getLinkState(robot, EE, computeForwardKinematics=True)
    return np.array(s[4]), s[5]


def eye_in_hand(robot):
    """Render the gripper-mounted downward camera -- the VLM's actual input."""
    pos, orn = ee_pose(robot)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    fwd, up = rot @ np.array([0, 0, 1]), rot @ np.array([0, 1, 0])
    eye = pos + fwd * 0.05
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
    _, _, rgba, _, _ = p.getCameraImage(CAM, CAM, view, proj, renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.reshape(rgba, (CAM, CAM, 4))[:, :, :3].astype("uint8"))


def scene_shot():
    """Render the external context camera used on the right side of the video."""
    _, _, rgba, _, _ = p.getCameraImage(SCN_W, SCN_H, SCENE_VIEW, SCENE_PROJ,
                                        renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.reshape(rgba, (SCN_H, SCN_W, 4))[:, :, :3].astype("uint8"))


def grab(robot):
    """Compose one output frame: camera feed + crosshair | scene inset + text."""
    canvas = Image.new("RGB", (W, H), (17, 21, 29))
    d = ImageDraw.Draw(canvas)

    # ---- top banner ----
    d.rectangle([0, 0, W, 50], fill=(11, 14, 20))
    d.text((18, 6), ui["title"], font=FT, fill=(255, 255, 255))
    d.text((18, 30), ui["state"], font=FM, fill=ui["col"])

    # ---- LEFT: eye-in-hand camera ----
    cam = eye_in_hand(robot)
    cx0, cy0 = 18, 70
    canvas.paste(cam, (cx0, cy0))
    d.rectangle([cx0 - 1, cy0 - 1, cx0 + CAM, cy0 + CAM], outline=(60, 74, 92), width=1)
    d.text((cx0, cy0 - 18), "EYE-IN-HAND CAMERA  \u2014  what the VLM sees",
           font=FL, fill=(230, 130, 60))

    # crosshair at image centre = the gripper's current target
    mx, my = cx0 + CAM // 2, cy0 + CAM // 2
    ch = (255, 90, 60)
    for dx in (-1, 0, 1):
        d.line([mx + dx, my - 26, mx + dx, my - 8], fill=ch)
        d.line([mx + dx, my + 8, mx + dx, my + 26], fill=ch)
        d.line([mx - 26, my + dx, mx - 8, my + dx], fill=ch)
        d.line([mx + 8, my + dx, mx + 26, my + dx], fill=ch)
    d.ellipse([mx - 5, my - 5, mx + 5, my + 5], outline=ch, width=2)
    d.text((mx + 30, my - 8), "gripper target", font=FM, fill=ch)

    # ---- RIGHT: scene inset ----
    sx0, sy0 = CAM + 44, 70
    canvas.paste(scene_shot(), (sx0, sy0))
    d.rectangle([sx0 - 1, sy0 - 1, sx0 + SCN_W, sy0 + SCN_H], outline=(60, 74, 92), width=1)
    d.text((sx0, sy0 - 18), "SCENE VIEW  \u2014  context", font=FL, fill=(120, 145, 175))

    # ---- RIGHT: VLM diagnosis panel ----
    py = sy0 + SCN_H + 22
    d.rectangle([sx0, py, sx0 + SCN_W, py + 148], fill=(11, 14, 20),
                outline=(45, 58, 74))
    d.text((sx0 + 14, py + 10), "VLM DIAGNOSIS", font=FL, fill=(120, 145, 175))
    yy = py + 34
    for line in ui["diag"].split("\n"):
        d.text((sx0 + 14, yy), line, font=FB, fill=(215, 228, 242))
        yy += 22

    frames.append(np.asarray(canvas))


def hstep(robot, n=1):
    """Advance physics and keep a successfully grasped cube attached to the gripper."""
    for _ in range(n):
        p.stepSimulation()
        if held["on"]:
            e, _ = ee_pose(robot)
            p.resetBasePositionAndOrientation(
                held["obj"], [e[0], e[1], e[2] - 0.045],
                p.getQuaternionFromEuler([0, 0, held["yaw"]]))


def move(robot, xyz, yaw, steps=48, cap=6):
    """Move the arm to a Cartesian pose with a specified planar yaw."""
    jt = p.calculateInverseKinematics(robot, EE, list(xyz), yq(yaw))
    for j, t in zip(ARM, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for i in range(steps):
        hstep(robot)
        if i % cap == 0:
            grab(robot)


def hold(robot, steps=18):
    """Pause the scene while continuing to record video frames."""
    for i in range(steps):
        hstep(robot)
        if i % 3 == 0:
            grab(robot)


def fingers(robot, v, steps=12):
    """Open or close the gripper fingers while recording frames."""
    for j in FING:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, v, force=40)
    for i in range(steps):
        hstep(robot)
        if i % 4 == 0:
            grab(robot)


def build(xy, yaw):
    """Reset PyBullet and create one robot/object scene at a given pose."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(); p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(ARM, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in FING:
        p.resetJointState(robot, j, F_OPEN)
    obj = p.loadURDF("cube_small.urdf", [xy[0], xy[1], 0.02],
                     p.getQuaternionFromEuler([0, 0, yaw]))
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    for _ in range(40):
        p.stepSimulation()
    return robot, obj


def try_grasp(robot, obj, axy, ayaw, oxy, oyaw):
    """Attempt a grasp and check both position and yaw alignment."""
    held["on"] = False
    move(robot, [axy[0], axy[1], HOVER_Z], ayaw)
    move(robot, [axy[0], axy[1], 0.055], ayaw)
    ok = (np.linalg.norm(np.array(axy) - np.array(oxy)) < CAPTURE
          and abs(math.degrees(ayaw - oyaw)) < YAW_TOL)
    fingers(robot, F_CLOSE if ok else F_OPEN)
    if ok:
        held.update(on=True, obj=obj, yaw=ayaw)
    move(robot, [axy[0], axy[1], 0.27], ayaw)
    hold(robot, 8)
    return ok


def mock_diagnose(axy, ayaw, oxy, oyaw, k=3):
    """Median-aggregated estimate over k frames + a confidence, as the FSM uses."""
    tp = (np.array(oxy) - np.array(axy)) * 100.0
    ty = math.degrees(oyaw - ayaw)
    pd = [tp + np.random.normal(0, 1.3, 2) for _ in range(k)]
    est = np.median(np.array(pd), axis=0)
    eyaw = float(np.median([ty + np.random.normal(0, 4.0) for _ in range(k)]))
    spread = float(np.mean(np.std(np.array(pd), axis=0)))
    conf = float(np.clip(0.95 - 0.09 * 1.3 - 0.02 * float(np.linalg.norm(tp))
                         + 0.07 * (k - 1) / (1 + spread), 0.05, 0.99))
    return est[0], est[1], eyaw, conf


# (object_xy, object_yaw_deg, sabotage(dx,dy,dyaw_deg), label)
SCENARIOS = [
    ((0.50,  0.00),   0, ( 0.040, -0.018,  14), "centre  \u00b7  offset RIGHT"),
    ((0.45,  0.15),  28, (-0.020,  0.038, -18), "back-left  \u00b7  offset FORWARD"),
    ((0.56, -0.13), -22, ( 0.034,  0.026,  20), "front-right  \u00b7  offset DIAGONAL"),
    ((0.53,  0.09),  35, (-0.038, -0.020, -16), "far-centre  \u00b7  offset BACK-LEFT"),
]

AMBER, BLUE, GREEN = (240, 205, 100), (150, 200, 255), (110, 220, 130)


def main():
    """Render all camera-view recovery scenarios into the MP4 output."""
    p.connect(p.DIRECT)
    recovered = 0
    for i, (oxy, oyaw_d, sab, label) in enumerate(SCENARIOS, 1):
        oyaw = math.radians(oyaw_d)
        robot, obj = build(oxy, oyaw)
        ui["title"] = f"Scenario {i}/{len(SCENARIOS)}   \u2014   {label}"

        aim = (oxy[0] + sab[0], oxy[1] + sab[1])
        ayaw = oyaw + math.radians(sab[2])

        # --- attempt (sabotaged) ---
        ui["col"] = AMBER
        ui["state"] = (f"FSM: APPROACH \u2192 GRASP   |   object at "
                       f"({oxy[0]:.2f}, {oxy[1]:+.2f}) m, yaw {oyaw_d:+d}\u00b0")
        ui["diag"] = ("saboteur injected:\n"
                      f"  {math.hypot(sab[0], sab[1])*100:.1f} cm offset, {sab[2]:+d}\u00b0 yaw\n"
                      "watch: object sits AWAY\nfrom the crosshair \u2192 miss")
        try_grasp(robot, obj, aim, ayaw, oxy, oyaw)

        ui["state"] = "FSM: CHECK \u2192 MISS.  Lifting to diagnostic hover..."
        ui["diag"] = ("grasp FAILED\n\nopen-loop would stop here\n(binary fault code)")
        move(robot, [aim[0], aim[1], HOVER_Z], ayaw)
        hold(robot, 14)

        # --- diagnose from the camera frame ---
        ui["col"] = BLUE
        ui["state"] = "FSM: DIAGNOSE   |   sending 3 temporal frames to the VLM"
        ui["diag"] = ("capturing eye-in-hand frames...\n\nthe object visible off-crosshair\nIS the error being measured")
        hold(robot, 20)

        dx, dy, dth, conf = mock_diagnose(np.array(aim), ayaw, oxy, oyaw)
        ui["state"] = f"CONFIDENCE GATE: {conf:.2f} \u2265 0.60  \u2192  ACT"
        ui["diag"] = (f"dx = {dx:+.1f} cm\n"
                      f"dy = {dy:+.1f} cm\n"
                      f"d\u03b8 = {dth:+.0f}\u00b0\n"
                      f"confidence = {conf:.2f}")
        hold(robot, 22)

        # --- recover ---
        ui["col"] = GREEN
        ui["state"] = "FSM: RECOVER   |   applying correction and retrying"
        cxy = (aim[0] + dx / 100.0, aim[1] + dy / 100.0)
        cyaw = ayaw + math.radians(dth)
        ok = try_grasp(robot, obj, cxy, cyaw, oxy, oyaw)
        recovered += int(ok)
        ui["state"] = f"RECOVERED   \u2014   object now centred on the crosshair   [{recovered}/{i}]"
        ui["diag"] = ("correction applied\n\nobject is now ON the crosshair\n\u2192 grasped and lifted")
        hold(robot, 26)

    p.disconnect()
    imageio.mimsave(OUT_MP4, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {OUT_MP4}  ({len(frames)} frames, {len(frames)/FPS:.1f}s, "
          f"recovered {recovered}/{len(SCENARIOS)})")


if __name__ == "__main__":
    main()
