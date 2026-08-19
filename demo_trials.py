"""
BATCH TRIALS: VLM failure-recovery across DIFFERENT LOCATIONS and ORIENTATIONS.

60 randomized trials. Each trial: an object at a random workspace (x,y) and a
random yaw. The saboteur induces BOTH a positional offset AND a yaw offset, so the
gripper misses in position and mis-aligns in rotation. The VLM recovery returns a
3-part correction (dx, dy, dtheta); success requires the gripper to land within the
capture radius AND within the yaw tolerance. Each trial is run under open-loop
(baseline) and closed-loop (confidence-gated, 3-frame temporal) for a paired
comparison.

Real PyBullet + Franka Panda + yaw-aligned top-down grasp. VLM is the labelled mock
(RealVLM one flag away). Outputs: console summary, CSV log, workspace map, summary
chart.
"""
import os, csv, math, random, time
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data

OUT = Path(__file__).resolve().parent / "outputs"
_FRAMES = OUT / "_frames"
OUT.mkdir(exist_ok=True)
_FRAMES.mkdir(exist_ok=True)

# ---- VLM backend: flip to True to run these trials against real Llama 3.2 Vision ----
USE_REAL_VLM = False
VLM_MODEL = "llama3.2-vision"
# sign calibration for the real camera (flip once if a correction worsens the miss)
CAM_DX_SIGN, CAM_DY_SIGN, CAM_DTH_SIGN = +1.0, +1.0, +1.0

try:
    import ollama
    _OLLAMA = True
except Exception:
    _OLLAMA = False

EE = 11
ARM = [0, 1, 2, 3, 4, 5, 6]
FING = [9, 10]
F_OPEN = 0.04
DOWN = None

# workspace + task tolerances
X_RANGE = (0.40, 0.60)
Y_RANGE = (-0.20, 0.20)
YAW_RANGE_DEG = (-40, 40)
CAPTURE_RADIUS = 0.028          # m  (position)
YAW_TOL_DEG = 12.0              # deg (orientation)

# saboteur magnitudes
SAB_POS = 0.045                 # +/- m
SAB_YAW_DEG = 25.0              # +/- deg

# noise (realistic 1x) -- perception (VLM) and actuation (robot)
PERC_POS_CM, PERC_YAW_DEG = 2.3, 7.5
ACT_POS_CM, ACT_YAW_DEG = 1.1, 3.5

# advanced-loop settings
TEMPORAL_FRAMES = 3
CONF_ACT, CONF_REQUERY = 0.60, 0.35
MAX_RECOVERIES = 3
OUT_OF_FRAME_CM = 12.0

N_TRIALS = 120

F_POS, F_ORI, F_NO = "POSITIONAL_OFFSET", "ORIENTATION", "NO_OBJECT"


def yaw_to_quat(yaw):
    """Convert a planar yaw angle to the gripper-down quaternion."""
    return p.getQuaternionFromEuler([math.pi, 0, yaw])


def build(obj_xy, obj_yaw):
    """Create a fresh trial scene with the object at a given position and yaw."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(); p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for j, a in zip(ARM, [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]):
        p.resetJointState(robot, j, a)
    for j in FING:
        p.resetJointState(robot, j, F_OPEN)
    quat = p.getQuaternionFromEuler([0, 0, obj_yaw])
    obj = p.loadURDF("cube_small.urdf", [obj_xy[0], obj_xy[1], 0.02], quat)
    p.changeVisualShape(obj, -1, rgbaColor=[0.9, 0.35, 0.05, 1])
    for _ in range(40):
        p.stepSimulation()
    return robot, obj


def move(robot, xyz, yaw, steps=55):
    """Move the gripper to a Cartesian target with the requested yaw."""
    jt = p.calculateInverseKinematics(robot, EE, list(xyz), yaw_to_quat(yaw))
    for j, t in zip(ARM, jt[:7]):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, t, force=250)
    for _ in range(steps):
        p.stepSimulation()


def attempt(robot, aim_xy, aim_yaw, sab=None, act_pos_cm=ACT_POS_CM, act_yaw_deg=ACT_YAW_DEG):
    """Execute a grasp with optional position/yaw sabotage and actuation noise."""
    x, y = aim_xy; yaw = aim_yaw
    if sab is not None:
        x += sab[0]; y += sab[1]; yaw += sab[2]
    move(robot, [x, y, 0.25], yaw); move(robot, [x, y, 0.05], yaw)
    land_xy = np.array([x, y]) + np.random.normal(0, act_pos_cm / 100.0, 2)
    land_yaw = yaw + math.radians(np.random.normal(0, act_yaw_deg))
    return (np.array([x, y]), yaw), (land_xy, land_yaw)


def succeeded(land, obj_xy, obj_yaw):
    """Check whether both position and orientation are within success tolerance."""
    land_xy, land_yaw = land
    pos_ok = np.linalg.norm(land_xy - np.array(obj_xy)) < CAPTURE_RADIUS
    yaw_ok = abs(math.degrees(land_yaw - obj_yaw)) < YAW_TOL_DEG
    return pos_ok and yaw_ok


def classify(aim_xy, aim_yaw, obj_xy, obj_yaw):
    """Classify the failure type from positional and orientation error."""
    pos_cm = np.linalg.norm(np.array(obj_xy) - np.array(aim_xy)) * 100
    yaw_deg = abs(math.degrees(obj_yaw - aim_yaw))
    if pos_cm > OUT_OF_FRAME_CM:
        return F_NO
    if yaw_deg > 10 and pos_cm < 3.0:
        return F_ORI
    return F_POS


def capture_eih(robot, path, jitter=0.0):
    """Render an eye-in-hand frame (for the real VLM) and save it to `path`."""
    from PIL import Image
    s = p.getLinkState(robot, EE, computeForwardKinematics=True)
    pos = np.array(s[4]); rot = np.array(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)
    fwd = rot @ np.array([0, 0, 1]); up = rot @ np.array([0, 1, 0])
    eye = pos + fwd * 0.05 + np.array([jitter, jitter, 0])
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd * 0.5).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 2.0)
    w, h, rgba, _, _ = p.getCameraImage(320, 320, view, proj, renderer=p.ER_TINY_RENDERER)
    Image.fromarray(np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)).save(path)
    return path


REAL_PROMPT = (
    "You are a top-down pick-and-place failure-diagnosis module with a downward "
    "eye-in-hand camera. The image centre is the gripper target; +x is image-right, "
    "+y is image-up. The grasp missed the object in both position and orientation. "
    "Reason briefly, then output ONLY JSON of the form "
    '{"dx_cm":<n>,"dy_cm":<n>,"dtheta_deg":<n>,"confidence":<0-1>} giving the '
    "correction to ADD to the target to centre on and align with the object.")


class RealVLM:
    """Real Llama 3.2 Vision via Ollama. Sends the K temporal frames in one query
    and returns the 3-part correction. One flag (USE_REAL_VLM) switches it on."""
    def __init__(self, model=VLM_MODEL):
        """Verify Ollama is available and remember the model name."""
        if not _OLLAMA:
            raise RuntimeError("USE_REAL_VLM=True but the 'ollama' package is not "
                               "installed. Run: pip install ollama  (and install "
                               "Ollama for Windows, then `ollama pull llama3.2-vision`).")
        self.model = model

    def diagnose(self, aim_xy, aim_yaw, obj_xy, obj_yaw, k, frames=None):
        """Ask the real VLM for position/yaw correction from captured frames."""
        import json
        t0 = time.perf_counter()
        try:
            r = ollama.chat(model=self.model, format="json",
                            messages=[{"role": "system", "content": REAL_PROMPT},
                                      {"role": "user", "content": "Return the JSON.",
                                       "images": frames or []}],
                            options={"temperature": 0.1})
            d = json.loads(r["message"]["content"])
            return {"dx_cm": float(d["dx_cm"]), "dy_cm": float(d["dy_cm"]),
                    "dtheta_deg": float(d.get("dtheta_deg", 0.0)),
                    "confidence": float(d.get("confidence", 0.5)),
                    "_latency_s": time.perf_counter() - t0}
        except Exception as e:
            print("   [RealVLM] inference/parse failed:", e)
            # safe fallback: zero correction with zero confidence -> the gate aborts
            return {"dx_cm": 0.0, "dy_cm": 0.0, "dtheta_deg": 0.0, "confidence": 0.0,
                    "_latency_s": time.perf_counter() - t0}


def make_vlm():
    """Select the real or mock VLM backend from the global flag."""
    return RealVLM() if USE_REAL_VLM else MockVLM()


class MockVLM:
    """Returns dx_cm, dy_cm, dtheta_deg + confidence; temporal-aggregated.
    `frames` is accepted for interface parity with RealVLM and ignored."""
    def diagnose(self, aim_xy, aim_yaw, obj_xy, obj_yaw, k, frames=None):
        """Return a mock temporal correction for position and yaw error."""
        t0 = time.perf_counter()
        true_pos = (np.array(obj_xy) - np.array(aim_xy)) * 100
        true_yaw = math.degrees(obj_yaw - aim_yaw)
        pos_draws = [true_pos + np.random.normal(0, PERC_POS_CM, 2) for _ in range(k)]
        yaw_draws = [true_yaw + np.random.normal(0, PERC_YAW_DEG) for _ in range(k)]
        est_pos = np.median(np.array(pos_draws), axis=0)
        est_yaw = float(np.median(yaw_draws))
        spread = float(np.mean(np.std(np.array(pos_draws), axis=0))) if k > 1 else PERC_POS_CM
        off = float(np.linalg.norm(true_pos))
        conf = float(np.clip(0.95 - 0.09 * PERC_POS_CM - 0.02 * off
                             + 0.07 * (k - 1) / (1 + spread) + np.random.normal(0, 0.02),
                             0.05, 0.99))
        return {"dx_cm": float(est_pos[0]), "dy_cm": float(est_pos[1]),
                "dtheta_deg": est_yaw, "confidence": conf,
                "_latency_s": time.perf_counter() - t0}


def run_episode(robot, obj_xy, obj_yaw, sab, closed, vlm):
    """Run one paired trial episode in open-loop or closed-loop mode."""
    (aim_xy, aim_yaw), land = attempt(robot, obj_xy, obj_yaw, sab)
    if succeeded(land, obj_xy, obj_yaw):
        return {"success": True, "recoveries": 0, "ftype": "-"}
    if not closed:
        return {"success": False, "recoveries": 0, "ftype": "-"}

    tgt_xy = np.array(aim_xy, float); tgt_yaw = aim_yaw
    ftype_seen = F_POS
    for attempt_i in range(1, MAX_RECOVERIES + 1):
        move(robot, [tgt_xy[0], tgt_xy[1], 0.22], tgt_yaw)         # hover to view
        ftype = classify(tgt_xy, tgt_yaw, obj_xy, obj_yaw)
        ftype_seen = ftype
        if ftype == F_NO:                                         # widen search
            tgt_xy = np.array(obj_xy, float)                      # nominal re-acquire
            move(robot, [tgt_xy[0], tgt_xy[1], 0.22], tgt_yaw)
            ftype = classify(tgt_xy, tgt_yaw, obj_xy, obj_yaw)

        k = TEMPORAL_FRAMES
        frames = None
        if USE_REAL_VLM:                                         # render eye-in-hand frames
            frames = [capture_eih(robot, _FRAMES / f"f{attempt_i}_{j}.png",
                                  jitter=0.004 * j) for j in range(k)]
            frames = [str(frame) for frame in frames]
        d = vlm.diagnose(tgt_xy, tgt_yaw, obj_xy, obj_yaw, k, frames)
        conf = d["confidence"]
        if conf < CONF_REQUERY:                                  # gate: abort
            return {"success": False, "recoveries": attempt_i, "ftype": ftype_seen,
                    "aborted": True}
        if conf < CONF_ACT:                                      # gate: re-query
            if USE_REAL_VLM:
                frames = [capture_eih(robot, _FRAMES / f"rq{attempt_i}_{j}.png",
                                      jitter=0.004 * j) for j in range(max(k, TEMPORAL_FRAMES))]
                frames = [str(frame) for frame in frames]
            d = vlm.diagnose(tgt_xy, tgt_yaw, obj_xy, obj_yaw, max(k, TEMPORAL_FRAMES), frames)
            conf = d["confidence"]
            if conf < CONF_REQUERY:
                return {"success": False, "recoveries": attempt_i, "ftype": ftype_seen,
                        "aborted": True}
        # act: apply 3-part correction (sign calibration matters for the real camera)
        tgt_xy = tgt_xy + np.array([CAM_DX_SIGN * d["dx_cm"], CAM_DY_SIGN * d["dy_cm"]]) / 100.0
        tgt_yaw = tgt_yaw + math.radians(CAM_DTH_SIGN * d["dtheta_deg"])
        (aim_xy, aim_yaw), land = attempt(robot, tuple(tgt_xy), tgt_yaw, None)
        if succeeded(land, obj_xy, obj_yaw):
            return {"success": True, "recoveries": attempt_i, "ftype": ftype_seen}
    return {"success": False, "recoveries": MAX_RECOVERIES, "ftype": ftype_seen}


def main():
    """Run randomized trials and save CSV plus summary charts."""
    global DOWN
    p.connect(p.DIRECT)
    DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
    np.random.seed(42); random.seed(42)
    vlm = make_vlm()
    print(f"VLM backend: {'RealVLM (Llama 3.2 Vision via Ollama)' if USE_REAL_VLM else 'MockVLM'}")
    rng = random.Random(42)

    rows = []
    print("=" * 78)
    print(f"BATCH: {N_TRIALS} VLM-recovery trials across locations & orientations")
    print("=" * 78)
    print(f"{'#':>3} {'x':>5} {'y':>6} {'yaw°':>5} | {'sab_pos_cm':>10} {'sab_yaw°':>8} | "
          f"{'open':>5} {'closed':>6} {'atts':>4} {'type':>18}")

    for i in range(1, N_TRIALS + 1):
        ox = rng.uniform(*X_RANGE); oy = rng.uniform(*Y_RANGE)
        oyaw = math.radians(rng.uniform(*YAW_RANGE_DEG))
        sab = (rng.uniform(-SAB_POS, SAB_POS), rng.uniform(-SAB_POS, SAB_POS),
               math.radians(rng.uniform(-SAB_YAW_DEG, SAB_YAW_DEG)))
        sab_pos_cm = math.hypot(sab[0], sab[1]) * 100
        sab_yaw_deg = abs(math.degrees(sab[2]))

        robot, obj = build((ox, oy), oyaw)
        r_open = run_episode(robot, (ox, oy), oyaw, sab, closed=False, vlm=vlm)
        robot, obj = build((ox, oy), oyaw)
        r_closed = run_episode(robot, (ox, oy), oyaw, sab, closed=True, vlm=vlm)

        rows.append({"trial": i, "x": round(ox, 3), "y": round(oy, 3),
                     "yaw_deg": round(math.degrees(oyaw), 1),
                     "sab_pos_cm": round(sab_pos_cm, 1), "sab_yaw_deg": round(sab_yaw_deg, 1),
                     "open_success": r_open["success"], "closed_success": r_closed["success"],
                     "recoveries": r_closed["recoveries"], "failure_type": r_closed["ftype"],
                     "aborted": r_closed.get("aborted", False)})
        print(f"{i:>3} {ox:5.2f} {oy:+6.2f} {math.degrees(oyaw):5.0f} | "
              f"{sab_pos_cm:10.1f} {sab_yaw_deg:8.0f} | "
              f"{str(r_open['success']):>5} {str(r_closed['success']):>6} "
              f"{r_closed['recoveries']:>4} {r_closed['ftype']:>18}")

    p.disconnect()

    # ---- summary ----
    n = len(rows)
    o_succ = sum(r["open_success"] for r in rows)
    c_succ = sum(r["closed_success"] for r in rows)
    aborts = sum(r["aborted"] for r in rows)
    mean_rec = np.mean([r["recoveries"] for r in rows if r["closed_success"]])
    from collections import Counter
    types = Counter(r["failure_type"] for r in rows if r["failure_type"] != "-")

    def wilson(k, m, z=1.96):
        """95% Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
        if m == 0:
            return 0.0, 0.0, 0.0
        ph = k / m
        denom = 1 + z * z / m
        center = (ph + z * z / (2 * m)) / denom
        half = z * math.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / denom
        return ph, max(0.0, center - half), min(1.0, center + half)

    po, olo, ohi = wilson(o_succ, n)
    pc, clo, chi = wilson(c_succ, n)

    print("\n" + "=" * 78)
    print(f"RESULTS over {n} trials (all at distinct locations & orientations)")
    print("=" * 78)
    print(f"  Open-loop (baseline)  : {o_succ:>3}/{n} = {100*po:5.1f}%  "
          f"(95% CI {100*olo:.1f}-{100*ohi:.1f}%)")
    print(f"  Closed-loop (recovery): {c_succ:>3}/{n} = {100*pc:5.1f}%  "
          f"(95% CI {100*clo:.1f}-{100*chi:.1f}%)")
    print(f"  Improvement           : +{100*(pc-po):.1f} percentage points")
    print(f"  Mean recovery attempts (succ): {mean_rec:.2f}")
    print(f"  Safe aborts (low confidence) : {aborts}")
    print(f"  Failure types encountered    : {dict(types)}")

    # ---- CSV ----
    with open(OUT / "trials_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- charts ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # workspace map: each object as an arrow (location + yaw), coloured by recovery
    fig, ax = plt.subplots(figsize=(7.4, 6))
    for r in rows:
        yaw = math.radians(r["yaw_deg"])
        col = "#3b7dc0" if r["closed_success"] else "#c0563b"
        ax.quiver(r["x"], r["y"], math.cos(yaw), math.sin(yaw),
                  color=col, angles="xy", scale=22, width=0.006, alpha=0.9)
        if not r["open_success"]:
            ax.scatter(r["x"], r["y"], s=140, facecolors="none",
                       edgecolors="#999", linewidths=0.8, zorder=0)
    ax.set_xlim(X_RANGE[0] - 0.03, X_RANGE[1] + 0.03)
    ax.set_ylim(Y_RANGE[0] - 0.05, Y_RANGE[1] + 0.05)
    ax.set_xlabel("workspace x (m)"); ax.set_ylabel("workspace y (m)")
    ax.set_title(f"{n} trials: object location + orientation (arrow = yaw)\n"
                 f"blue = recovered, red = failed, grey ring = open-loop also failed")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "workspace_map.png", dpi=130)

    # summary bars (with 95% Wilson CI error bars) + recovery-attempts histogram
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    rates = [100 * po, 100 * pc]
    yerr = [[100 * (po - olo), 100 * (pc - clo)],      # lower
            [100 * (ohi - po), 100 * (chi - pc)]]      # upper
    a1.bar(["Open-loop", "Closed-loop"], rates,
           color=["#c0563b", "#3b7dc0"], width=0.55)
    a1.errorbar([0, 1], rates, yerr=yerr, fmt="none", ecolor="black",
                capsize=7, capthick=1.5, elinewidth=1.5)
    for i, (v, lo, hi) in enumerate([(rates[0], olo, ohi), (rates[1], clo, chi)]):
        a1.text(i, min(v + 6, 99), f"{v:.0f}%", ha="center", fontweight="bold")
    a1.set_ylim(0, 105); a1.set_ylabel("success rate (%)")
    a1.set_title(f"Success over {n} trials (95% CI error bars)")
    a1.grid(axis="y", alpha=0.3)

    recs = [r["recoveries"] for r in rows if r["closed_success"]]
    a2.hist(recs, bins=[0.5, 1.5, 2.5, 3.5], color="#3b7dc0", rwidth=0.7)
    a2.set_xticks([1, 2, 3]); a2.set_xlabel("recovery attempts to succeed")
    a2.set_ylabel("number of trials"); a2.set_title("Attempts needed to recover")
    a2.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "trials_summary.png", dpi=130)

    print(f"\n  wrote trials_results.csv, workspace_map.png, trials_summary.png")


if __name__ == "__main__":
    main()
