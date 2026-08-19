"""
probe_vlm.py - sanity-check and CALIBRATE Llama 3.2 Vision before running trials.

It renders ONE eye-in-hand frame of a cube deliberately offset from the gripper
centre, sends it to the real model via Ollama, and prints the raw JSON correction.
Use it to (a) confirm Ollama + the model work, (b) see the real latency, and
(c) CALIBRATE THE SIGNS: the printed dx_cm/dy_cm should point FROM the gripper
centre TOWARD the object. If a sign is inverted, flip the matching CAM_*_SIGN in
demo.py / demo_advanced.py / demo_trials.py.

Run:  python probe_vlm.py
Needs: Ollama running + `ollama pull llama3.2-vision` + `pip install ollama pybullet pillow numpy`
"""
import json, time, math
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

MODEL = "llama3.2-vision"
# put the object at this offset (metres) relative to where the gripper is aiming.
# +x = world x (roughly image-right for our top-down cam), +y = world y.
TRUE_OFFSET = (0.04, -0.02)     # object is 4cm +x and 2cm -y from the gripper centre
TRUE_YAW_DEG = 15.0             # object rotated 15 deg

PROMPT = (
    "You are a top-down pick-and-place failure-diagnosis module with a downward "
    "eye-in-hand camera. The image centre is the gripper target; +x is image-right, "
    "+y is image-up. The grasp missed. Reason briefly, then output ONLY JSON "
    '{"dx_cm":<n>,"dy_cm":<n>,"dtheta_deg":<n>,"confidence":<0-1>} giving the '
    "correction to ADD to the target to centre on and align with the object.")


def render_frame(path):
    """Render a simple eye-in-hand test image for probing the real VLM."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(range(7), [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    # gripper aims at (0.5, 0.0); object sits at aim + TRUE_OFFSET, rotated
    aim = np.array([0.5, 0.0])
    obj_xy = aim + np.array(TRUE_OFFSET)
    quat = p.getQuaternionFromEuler([0, 0, math.radians(TRUE_YAW_DEG)])
    obj = p.loadURDF("cube_small.urdf", [obj_xy[0], obj_xy[1], 0.02], quat)
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    # move gripper to hover over the (wrong) aim so the object is off-centre in view
    down = p.getQuaternionFromEuler([math.pi, 0, 0])
    jt = p.calculateInverseKinematics(robot, 11, [aim[0], aim[1], 0.22], down)
    for j, t in zip(range(7), jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for _ in range(120):
        p.stepSimulation()
    s = p.getLinkState(robot, 11, computeForwardKinematics=True)
    pos = np.array(s[4]); rot = np.array(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)
    fwd = rot @ np.array([0, 0, 1]); up = rot @ np.array([0, 1, 0])
    eye = pos + fwd * 0.05
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
    w, h, rgba, _, _ = p.getCameraImage(320, 320, view, proj, renderer=p.ER_TINY_RENDERER)
    Image.fromarray(np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)).save(path)
    p.disconnect()


def main():
    """Render one probe image, send it to Ollama, and print the VLM JSON."""
    import ollama
    frame = "probe_frame.png"
    render_frame(frame)
    print(f"Rendered {frame}. True object offset from gripper: "
          f"dx={TRUE_OFFSET[0]*100:+.1f}cm dy={TRUE_OFFSET[1]*100:+.1f}cm "
          f"yaw={TRUE_YAW_DEG:+.0f}deg")
    print("Querying Llama 3.2 Vision (first call may be slow while the model loads)...")
    t0 = time.perf_counter()
    r = ollama.chat(model=MODEL, format="json",
                    messages=[{"role": "system", "content": PROMPT},
                              {"role": "user", "content": "Return the JSON.",
                               "images": [frame]}],
                    options={"temperature": 0.1})
    dt = time.perf_counter() - t0
    print(f"\nlatency: {dt:.2f}s")
    print("raw model reply:", r["message"]["content"])
    try:
        d = json.loads(r["message"]["content"])
        print(f"\nparsed: dx={d.get('dx_cm')}  dy={d.get('dy_cm')}  "
              f"dtheta={d.get('dtheta_deg')}  conf={d.get('confidence')}")
        print("\nCALIBRATION CHECK:")
        print(f"  expected dx sign: + (object is +{TRUE_OFFSET[0]*100:.0f}cm)  "
              f"-> model gave {d.get('dx_cm')}")
        print(f"  expected dy sign: - (object is {TRUE_OFFSET[1]*100:.0f}cm)   "
              f"-> model gave {d.get('dy_cm')}")
        print("  If a sign is inverted, flip the matching CAM_*_SIGN in the trial scripts.")
    except Exception as e:
        print("Could not parse JSON:", e)


if __name__ == "__main__":
    main()
