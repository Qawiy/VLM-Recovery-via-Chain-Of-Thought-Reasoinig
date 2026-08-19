Autonomous Failure Recovery in Robotic Manipulation Using Vision-Language Models
Research project investigating VLM-based failure detection, diagnosis
and recovery for robotic manipulation.
Overview
This project implements a closed-loop robotic manipulation recovery
system combining:
a Vision-Language Model (VLM) cognitive layer;
a confidence-gated Finite State Machine (FSM);
a PyBullet physics simulation;
a 7-DOF Franka Panda manipulator;
an eye-in-hand 320×320 RGB-D camera;
locally served qwen2.5vl through Ollama.
The system observes a failed top-down grasp, classifies the failure,
asks the VLM to diagnose the visual error, evaluates its confidence,
transforms the proposed correction into the robot frame, and retries the
grasp when the diagnosis is sufficiently reliable.
> **Status:** Academic research prototype. The reported validation is
> simulation-based and does not establish physical-robot deployment.
Research Objectives
The project aims to:
Build a reproducible PyBullet simulation of a 7-DOF Franka Panda.
Inject controlled positional and rotational grasp failures.
Use a VLM to estimate object position, orientation and corrective
motion.
Implement confidence-gated recovery through an FSM.
Investigate temporal multi-frame VLM queries.
Route prompts according to failure type.
Evaluate robustness using a sim-to-real noise ablation.
Record semantic fault information for recovery decisions.
Compare closed-loop recovery with an open-loop baseline over 120
paired trials.
System Architecture
``` text
                 ┌─────────────────────────┐
                 │      COGNITIVE LAYER     │
                 │                         │
                 │ Eye-in-hand image       │
                 │          ↓              │
                 │      qwen2.5vl           │
                 │          ↓              │
                 │ Diagnosis + correction  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      EXECUTION LAYER     │
                 │                         │
                 │ Failure classification │
                 │          ↓              │
                 │ Confidence-gated FSM   │
                 │          ↓              │
                 │ Coordinate transform   │
                 │          ↓              │
                 │ Correction + retry     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     SIMULATION LAYER     │
                 │                         │
                 │ PyBullet               │
                 │ Franka Panda           │
                 │ RGB-D camera            │
                 │ Target cube             │
                 │ Failure injection       │
                 └─────────────────────────┘
```
The implementation is organised around a world builder, grasp primitive,
VLM backend, confidence-gated FSM, camera-to-base coordinate transform
and semantic logger.
Key Features
Confidence-Gated Recovery
The VLM confidence controls the next action:
Confidence     Action
---
`≥ 0.60`       Accept diagnosis and recover
`0.35–<0.60`   Re-query using additional frames
`< 0.35`       Abort and log
Temporal Multi-Frame Diagnosis
The system supports a three-frame temporal window. Multiple observations
can be aggregated to reduce the influence of individual noisy estimates.
Failure-Type Prompt Routing
Failures are classified as:
`POSITIONAL_OFFSET`
`ORIENTATION`
`NO_OBJECT`
The classification determines the diagnostic prompt and recovery
pathway.
Sim-to-Real Noise Ablation
Perception and actuation noise are swept from a clean baseline to three
times the nominal level. Open-loop, single-frame closed-loop and
three-frame temporal closed-loop strategies are compared.
Semantic Fault Logging
Recovery decisions can be stored as structured JSON containing the
failure type, confidence, correction vector, outcome and model reasoning
information.
Simulation Configuration
Parameter            Value
---
Simulator            PyBullet
Environment          Native Windows
Robot                Franka Panda
Robot DOF            7
Camera               320×320 RGB-D
Camera               Eye-in-hand, top-down
VLM                  qwen2.5vl
VLM runtime          Ollama
Correction           `dx`, `dy`, `dθ`
Temporal window      3 frames
Act threshold        0.60
Abort threshold      0.35
Position tolerance   2.8 cm
Yaw tolerance        12°
Main evaluation      120 paired trials
The validated implementation uses native-Windows PyBullet as a
single-process simulation. The earlier design considered Webots, ROS 2
and MoveIt 2, but the experimental implementation was simplified to
PyBullet to reduce middleware integration overhead. ROS 2 and MoveIt 2
remain a proposed route for future physical-robot validation.
Failure Injection
A scripted saboteur introduces controlled failures using:
planar positional offsets of up to approximately 4.5 cm;
yaw disturbances of up to approximately 25°.
A trial is considered successful when the final gripper position is
within 2.8 cm of the target and within 12° of its yaw.
Recovery Workflow
``` text
APPROACH
   ↓
GRASP
   │
   ├── Success → DONE
   │
   └── Failure
          ↓
       CLASSIFY
          ↓
       DIAGNOSE
          │
          ├── confidence ≥ 0.60 → RECOVER → GRASP
          │
          ├── 0.35–<0.60 → RE-QUERY
          │
          └── <0.35 → ABORT
```
Recovery attempts are bounded to prevent uncontrolled repeated motion.
VLM Interface
The VLM is prompted to locate the cube, estimate its orientation,
determine a correction and provide confidence.
Expected structured output:
``` json
{
  "reasoning": "...",
  "dx_cm": 3.4,
  "dy_cm": -1.8,
  "dtheta_deg": -15,
  "confidence": 0.82
}
```
Because the grasp is top-down, the correction is represented using two
translational variables and one yaw variable.
Experimental Results
The principal live-model evaluation used 120 trials.
Strategy                    Success rate
---
Open-loop                      11.7%
Closed-loop + qwen2.5vl        18.3%
The closed-loop system improved the observed success rate by
approximately 6.7 percentage points.
The reported 95% Wilson confidence intervals were:
Open-loop: 7.1--18.6%
Closed-loop: 12.4--26.2%
The intervals overlap; therefore, the live-model improvement is reported
as positive but not statistically significant under the project's
strict confidence-interval non-overlap criterion.
Failure Breakdown
The 102 classified failures comprised:
Failure                Count
---
Positional offset     49
Orientation           29
No object             24
Improving the eye-in-hand view reduced `NO_OBJECT` failures from
approximately 35% to 20%. The remaining bottleneck therefore shifted
toward fine-grained spatial estimation: the model can often
recognise the qualitative direction of the displacement but has
difficulty producing a sufficiently accurate metric correction.
Noise Ablation
The noise experiment evaluates robustness rather than clean-room peak
performance. It shows that temporal reasoning can be beneficial under
moderate noise, but performance is not monotonic at the highest noise
level. When all frames become strongly corrupted, aggregating several
observations cannot reliably remove the underlying error.
Recommended Repository Structure
Use the actual files in the repository when applying this structure:
``` text
robot-vlm-failure-recovery/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── simulation/
│   ├── robot/
│   ├── camera/
│   ├── vlm/
│   ├── fsm/
│   ├── recovery/
│   ├── transforms/
│   └── logging/
├── prompts/
├── experiments/
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
├── data/
└── docs/
```
Installation
1. Clone the repository
``` bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```
2. Create a Python environment
Windows:
``` powershell
python -m venv .venv
.venv\Scripts	ctivate
```
Linux/macOS:
``` bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
If the repository contains `requirements.txt`:
``` bash
pip install -r requirements.txt
```
4. Install and configure Ollama
Install Ollama and make sure the local service is available.
Then obtain the model used in the project:
``` bash
ollama pull qwen2.5vl
```
Verify:
``` bash
ollama list
```
Follow the current Ollama documentation if commands differ for your
installed version.
Running the Project
Use the actual entry-point scripts contained in the repository. A
typical workflow is:
``` bash
.venv\Scripts	ctivate
ollama list
python <main_simulation_script>.py
```
For an experiment:
``` bash
python <experiment_script>.py
```
For evaluation:
``` bash
python <evaluation_script>.py
```
Replace the placeholders with the scripts included in the repository.
Reproducibility
To reproduce the reported experiments, keep the following consistent:
Franka Panda configuration;
camera configuration;
target object;
failure-injection ranges;
2.8 cm / 12° success criterion;
qwen2.5vl configuration;
confidence thresholds;
three-frame temporal window;
retry limit;
trial count;
evaluation procedure.
The VLM backend is separated from the recovery architecture so that the
idealised/mock and live configurations can use the same FSM, transform,
routing and logging interfaces.
Limitations
The project has several important limitations:
Validation is simulation-based; physical transfer has not been
established.
Grasp success uses a geometric capture criterion rather than
complete contact and friction physics.
Fine-grained VLM spatial estimation remains the main live-model
bottleneck.
The failure taxonomy is limited to three predefined visual
categories.
Simulated noise is a proxy for, rather than a replacement for, the
physical sim-to-real gap.
Results are specific to the tested qwen2.5vl configuration and
should not automatically be generalised to other VLMs.
Future Work
The report identifies several directions:
Improved Spatial Grounding
Combine VLM reasoning with explicit image-space localisation and camera
geometry:
``` text
Image
  ↓
Pixel localisation
  ↓
Depth / camera geometry
  ↓
Metric offset
  ↓
VLM reasoning
  ↓
Robot correction
```
Stronger Local VLMs
The modular VLM interface allows alternative local models to be
evaluated without redesigning the recovery controller.
Expanded Failure Taxonomy
Future versions can investigate occlusion, collision, object slippage,
unexpected obstacles and multi-object interference.
Physical Robot Validation
The next major validation step is deployment on a physical Panda or
equivalent robot, with appropriate safety controls. The original design
identifies ROS 2 and MoveIt 2 as a potential bridge for this stage.
Physical Grasp Validation
A future implementation should supplement the current position/yaw
capture criterion with a physics-based or hardware-based lift/contact
success check.
Safety
This repository is for research and educational use.
Do not connect the VLM output directly to unrestricted physical
robot motion. Any physical deployment requires motion limits, collision
protection, emergency stopping, human supervision and hardware-specific
validation.
Citation
``` bibtex
@misc{afolabi2026autonomous,
  author       = {Afolabi, Olujuwon},
  title        = {Autonomous Failure Recovery in Robotic Manipulation Using Vision-Language Models and Chain-of-Thought Reasoning},
  year         = {2026},
  institution  = {Newcastle University},
  note         = {EEE8097 Individual Project}
}
```
Acknowledgements
This work was completed as part of the EEE8097 Individual Project at
Newcastle University.
Special thanks to Dr. Osama Abushafa for supervision and guidance
throughout the project.
License
This is an academic research project. Add an appropriate open-source
licence to the repository before redistributing the code for wider use.
---
Author: Olujuwon Afolabi  
Institution: Newcastle University  
Project: EEE8097 Individual Project  
Year: 2026
