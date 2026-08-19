"""
Render advanced recovery scenarios using project_all_in_one.py as the simulation
source of truth.

This is intentionally shaped like render_advanced_trace.py: it uses a fixed
PyBullet scene camera, burns FSM state into a top banner, shows MCS messages in a
bottom overlay, and writes one MP4. The difference is that the recovery outputs
are collected from project_all_in_one.py's VLM/FSM primitives instead of being
reimplemented locally.

Run:
    python "all in one render scenario.py"

Optional:
    python "all in one render scenario.py" --real-vlm
"""

from pathlib import Path
import argparse
import math
import os
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

import project_all_in_one as sim


np.random.seed(0)

W, H = 780, 520
FPS = 30
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_MP4 = OUT_DIR / "all_in_one_advanced_trace.mp4"

VIEW = p.computeViewMatrixFromYawPitchRoll([0.47, 0.0, 0.1], 1.2, 48, -33, 0, 2)
PROJ = p.computeProjectionMatrixFOV(56, W / H, 0.1, 3.0)

frames = []
title = ""
state = ""
mcs_lines = []
held = {"on": False, "obj": None, "yaw": 0.0}

GATE_COLOR = {
    "ACT": (90, 210, 120),
    "REQUERY": (240, 200, 90),
    "ABORT": (240, 110, 90),
    "": (150, 200, 255),
}


def _font(size, bold=False):
    """Load an overlay font, with a portable fallback."""
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


F_TITLE = _font(22, True)
F_STATE = _font(18, True)
F_MCS = _font(14)


class OverlayMCSLogger(sim.MCSLogger):
    """Collect project_all_in_one MCS records and mirror them to the video overlay."""

    def __init__(self):
        path = Path(tempfile.gettempdir()) / "all_in_one_video_mcs.jsonl"
        super().__init__(str(path), verbose=False)

    def emit(self, event, message, **fields):
        rec = super().emit(event, message, **fields)
        prefix = event.replace("_", " ")
        mcs_lines.append(f"{prefix}  {message}")
        return rec


def grab(gate=""):
    """Capture one simulation frame with the current FSM and MCS overlays."""
    _, _, rgba, _, _ = p.getCameraImage(W, H, VIEW, PROJ, renderer=p.ER_TINY_RENDERER)
    img = Image.fromarray(np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 58], fill=(16, 20, 28))
    d.text((14, 6), title, font=F_TITLE, fill=(255, 255, 255))
    d.text((14, 33), state, font=F_STATE, fill=GATE_COLOR.get(gate, (150, 200, 255)))

    ph = 96
    d.rectangle([0, H - ph, W, H], fill=(12, 14, 20))
    d.text((14, H - ph + 6), "MCS semantic fault log", font=F_MCS, fill=(120, 140, 170))
    for i, line in enumerate(mcs_lines[-4:]):
        d.text((14, H - ph + 26 + i * 17), line[:118], font=F_MCS, fill=(190, 205, 225))
    frames.append(np.asarray(img))


def step_hold(robot, n=1):
    """Advance physics and keep a successfully grasped cube attached to the gripper."""
    for _ in range(n):
        p.stepSimulation()
        if held["on"] and held["obj"] is not None:
            e, _ = sim.ee_pose(robot)
            p.resetBasePositionAndOrientation(
                held["obj"],
                [e[0], e[1], e[2] - 0.045],
                p.getQuaternionFromEuler([0, 0, held["yaw"]]),
            )


def vmove(robot, xyz, yaw=0.0, steps=54, cap=4, gate=""):
    """Move with project_all_in_one IK conventions and record progress frames."""
    jt = p.calculateInverseKinematics(
        robot, sim.PANDA_EE_LINK, list(xyz), sim.yaw_to_quat(yaw)
    )
    for joint, target in zip(sim.PANDA_ARM_JOINTS, jt[:7]):
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, target, force=250)
    for i in range(steps):
        step_hold(robot)
        if i % cap == 0:
            grab(gate)


def vfingers(robot, value, steps=15, gate=""):
    """Move the Panda fingers and record frames."""
    for joint in sim.PANDA_FINGER_JOINTS:
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, value, force=40)
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab(gate)


def vhold(robot, steps=18, gate=""):
    """Pause while continuing to record frames."""
    for i in range(steps):
        step_hold(robot)
        if i % 3 == 0:
            grab(gate)


def vgrasp(robot, obj, aim_xy, aim_yaw=0.0, gate=""):
    """Video-visible equivalent of project_all_in_one.attempt_grasp()."""
    held["on"] = False
    vfingers(robot, sim.FINGER_OPEN, 9, gate)
    vmove(robot, [aim_xy[0], aim_xy[1], 0.25], aim_yaw, gate=gate)
    vmove(robot, [aim_xy[0], aim_xy[1], 0.055], aim_yaw, gate=gate)

    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj)
    obj_xy = np.array(obj_pos[:2])
    obj_yaw = p.getEulerFromQuaternion(obj_orn)[2]
    land = (np.array(aim_xy), aim_yaw)
    ok = sim.succeeded(land, obj_xy, obj_yaw, use_orientation=False)

    vfingers(robot, 0.015 if ok else sim.FINGER_OPEN, 15, gate)
    if ok:
        held.update(on=True, obj=obj, yaw=aim_yaw)
    vmove(robot, [aim_xy[0], aim_xy[1], 0.27], aim_yaw, gate=gate)
    vhold(robot, 10, gate)
    return ok


def capture_frames_for_vlm(robot, tag, k):
    """Capture temporal eye-in-hand frames only when the real VLM backend is enabled."""
    if not sim.USE_REAL_VLM:
        return None
    frame_paths = []
    for idx in range(k):
        path = sim.capture_rgbd(
            robot,
            os.path.join(sim.FRAMES_DIR, f"video_{tag}_{idx}.png"),
            jitter=0.004 * idx,
        )
        frame_paths.append(path)
    return frame_paths


def render_case(case_index, total_cases, obj_xy, sabotage, temporal_frames,
                perc_pos_cm, act_pos_cm, label, vlm):
    """Run one all-in-one scenario and render its observed VLM/FSM output."""
    global title, state, mcs_lines

    held.update(on=False, obj=None, yaw=0.0)
    mcs_lines = []
    mcs = OverlayMCSLogger()
    robot, obj = sim.build_world(obj_xy, 0.0)
    aim_xy = np.array([obj_xy[0] + sabotage[0], obj_xy[1] + sabotage[1]], float)
    target_xy = aim_xy.copy()
    target_yaw = 0.0

    title = f"Case {case_index}/{total_cases}: {label}"
    state = "FSM: APPROACH -> GRASP  (project_all_in_one initial attempt)"
    vgrasp(robot, obj, tuple(aim_xy), target_yaw)
    state = "FSM: CHECK -> MISS. Entering all-in-one recovery."
    mcs.emit("GRASP_FAILURE", "Initial grasp missed; closed-loop recovery engaged.",
             failure_type="UNDIAGNOSED", resolved=False)
    vhold(robot, 16)

    for attempt in range(1, sim.MAX_RECOVERIES + 1):
        vmove(robot, [target_xy[0], target_xy[1], 0.22], target_yaw)
        frames_for_vlm = capture_frames_for_vlm(robot, f"case{case_index}_a{attempt}", temporal_frames)

        ftype, fconf = vlm.classify(
            target_xy, target_yaw, obj_xy, 0.0, False, frames_for_vlm
        )
        state = f"ROUTING: classified {ftype}  (conf {fconf:.2f})"
        vhold(robot, 10)

        if ftype == sim.F_NO:
            mcs.emit("FAILURE_ROUTED", "Object not in view; widening search to nominal pose.",
                     failure_type=ftype, attempt=attempt, recovery_action="WIDEN_SEARCH",
                     resolved=False)
            state = "ROUTING: NO_OBJECT -> WIDEN SEARCH"
            vhold(robot, 12)
            target_xy = np.array(obj_xy, float)
            vmove(robot, [target_xy[0], target_xy[1], 0.22], target_yaw)
            frames_for_vlm = capture_frames_for_vlm(
                robot, f"case{case_index}_a{attempt}_wide", temporal_frames
            )
            ftype, fconf = vlm.classify(
                target_xy, target_yaw, obj_xy, 0.0, False, frames_for_vlm
            )
            state = f"re-acquired: {ftype} (conf {fconf:.2f})"
            vhold(robot, 10)

        diag = vlm.diagnose(
            ftype, target_xy, target_yaw, obj_xy, 0.0, temporal_frames, False,
            perc_pos_cm, 0.0, frames_for_vlm
        )
        if diag is None:
            mcs.emit("RECOVERY_ABORTED", "VLM diagnosis failed; no correction available.",
                     failure_type=ftype, attempt=attempt, resolved=False)
            state = "CONFIDENCE GATE: diagnosis unavailable -> ABORT"
            vhold(robot, 28, gate="ABORT")
            return "diagnosis_failed"

        conf = diag["confidence"]
        if conf < sim.CONF_REQUERY:
            mcs.emit("RECOVERY_ABORTED",
                     f"Confidence {conf:.2f} below safe threshold; aborting to avoid unsafe move.",
                     failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                     recovery_action="ABORT_LOW_CONFIDENCE", resolved=False)
            state = f"CONFIDENCE GATE: {conf:.2f} < {sim.CONF_REQUERY:.2f} -> ABORT"
            vhold(robot, 34, gate="ABORT")
            return "low_conf_abort"

        if conf < sim.CONF_ACT:
            mcs.emit("DIAGNOSIS_REQUERIED",
                     f"Marginal confidence {conf:.2f}; re-querying with temporal frames.",
                     failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                     recovery_action="REQUERY")
            state = f"CONFIDENCE GATE: {conf:.2f} marginal -> RE-QUERY"
            vhold(robot, 20, gate="REQUERY")
            frames2 = capture_frames_for_vlm(
                robot, f"case{case_index}_a{attempt}_requery",
                max(temporal_frames, sim.TEMPORAL_FRAMES),
            )
            diag2 = vlm.diagnose(
                ftype, target_xy, target_yaw, obj_xy, 0.0,
                max(temporal_frames, sim.TEMPORAL_FRAMES), False,
                perc_pos_cm, 0.0, frames2 or frames_for_vlm
            )
            if diag2 is not None:
                diag, conf = diag2, diag2["confidence"]
                state = f"re-query -> conf {conf:.2f}"
                vhold(robot, 10, gate="REQUERY")
            if conf < sim.CONF_REQUERY:
                mcs.emit("RECOVERY_ABORTED", f"Still low confidence ({conf:.2f}); abort.",
                         failure_type=ftype, confidence=round(conf, 2), attempt=attempt,
                         recovery_action="ABORT_LOW_CONFIDENCE", resolved=False)
                state = f"CONFIDENCE GATE: {conf:.2f} still low -> ABORT"
                vhold(robot, 28, gate="ABORT")
                return "low_conf_abort"

        dx, dy = diag["dx_cm"], diag["dy_cm"]
        mag = float(np.linalg.norm(np.array(obj_xy) - target_xy) * 100)
        mcs.emit("CORRECTION_APPLIED",
                 f"{ftype}: object {mag:.1f}cm off; applying correction "
                 f"(dx={dx:+.1f}, dy={dy:+.1f}) cm and retrying.",
                 failure_type=ftype, confidence=round(conf, 2),
                 correction_cm={"dx": round(dx, 2), "dy": round(dy, 2)},
                 frames_used=diag.get("_frames", temporal_frames), attempt=attempt,
                 recovery_action="RETRY", resolved=False)
        state = (f"CONFIDENCE GATE: {conf:.2f} >= {sim.CONF_ACT:.2f} -> ACT  "
                 f"(dx={dx:+.1f} dy={dy:+.1f}cm)")
        vhold(robot, 14, gate="ACT")

        target_xy = target_xy + sim.camera_to_base(dx, dy)
        if act_pos_cm:
            target_xy = target_xy + np.random.normal(0, act_pos_cm / 100.0, 2)
        ok = vgrasp(robot, obj, tuple(target_xy), target_yaw, gate="ACT")
        if ok:
            mcs.emit("RECOVERY_SUCCEEDED", f"Grasp recovered on attempt {attempt}.",
                     failure_type=ftype, attempt=attempt, resolved=True)
            state = f"RECOVERED on attempt {attempt}"
            vhold(robot, 30, gate="ACT")
            return "recovered"

    mcs.emit("RECOVERY_EXHAUSTED",
             f"Unrecovered after {sim.MAX_RECOVERIES} attempts; escalating.",
             failure_type=ftype, resolved=False)
    state = "RECOVERY EXHAUSTED -> ESCALATE"
    vhold(robot, 30, gate="ABORT")
    return "exhausted"


CASES = [
    ((0.50, 0.00), (0.038, -0.022), 3, 1.2, 0.6,
     "positional miss -> temporal (3f) diagnose -> ACT -> recover"),
    ((0.50, 0.00), (0.030, 0.030), 3, 3.6, 0.8,
     "high noise -> marginal confidence -> RE-QUERY -> recover"),
    ((0.50, 0.00), (0.028, -0.030), 1, 6.0, 1.0,
     "very noisy single-frame -> low confidence -> safe ABORT"),
    ((0.50, 0.00), (0.150, 0.020), 3, 1.2, 0.6,
     "gross offset -> NO_OBJECT routing -> widen search -> recover"),
]


def main():
    """Render all advanced all-in-one cases into one MP4."""
    global title, state

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-vlm", action="store_true",
                        help="use project_all_in_one RealVLM/Ollama backend")
    parser.add_argument("--out", default=str(OUT_MP4), help="output MP4 path")
    args = parser.parse_args()

    sim.USE_REAL_VLM = bool(args.real_vlm)
    np.random.seed(0)
    p.connect(p.DIRECT)
    sim._DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
    vlm = sim.make_vlm()

    try:
        outcomes = []
        for i, (obj_xy, sabotage, k, perc_pos, act_pos, label) in enumerate(CASES, 1):
            outcome = render_case(i, len(CASES), obj_xy, sabotage, k, perc_pos, act_pos,
                                  label, vlm)
            outcomes.append(outcome)
            print(f"  case {i}: {label[:52]:52s} -> {outcome}")

        title = "All-in-one simulation recovery trace complete"
        state = f"outcomes: {', '.join(outcomes)}"
        if CASES:
            vhold(sim.build_world((0.50, 0.00), 0.0)[0], 30)
    finally:
        p.disconnect()

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {out_path}  ({len(frames)} frames, {len(frames) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
