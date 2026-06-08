# Synthetic Environment Impact on YOLO Generalization in Robotic Perception
## Full Project Scope & Agent Guidelines

---

## PROJECT IDENTITY

**Title:** Synthetic Environment Impact on Object Detection Generalization in Robotic Perception Systems  
**Owner:** Sahan (Final Year UG, Computer Engineering, University of Ruhuna)  
**Duration:** 6–7 weeks  
**Purpose:** CV portfolio piece + learning ROS2/ML skills  
**HPC:** Ubuntu 24.04.4 LTS, RTX 5090 (32GB VRAM), 62GB RAM, CUDA 13.1, 737GB free disk  
**Access:** SSH via ngrok tunnel  

---

## RESEARCH QUESTION

> How does the structure of a synthetic training environment affect object detection model generalization across unseen simulated scenarios in robotic perception?

---

## FINAL TECH STACK

| Category | Tool | Version |
|---|---|---|
| Robot Framework | ROS2 | Jazzy (Ubuntu 24.04 LTS match) |
| Simulation | Gazebo | Harmonic (pairs with ROS2 Jazzy) |
| Robot Model | TurtleBot3 | Waffle Pi |
| Object Detection | YOLOv8 | Ultralytics latest |
| Deep Learning | PyTorch | Latest stable with CUDA |
| Analysis | Python, NumPy, Pandas, Matplotlib, Seaborn | Latest |
| Dashboard | Streamlit | Latest |
| HPC Jobs | SLURM batch | Available on HPC |

> ⚠️ Do NOT suggest ROS2 Humble or Gazebo 11/Classic — those target Ubuntu 22.04. This system runs Ubuntu 24.04.

---

## 6 DETAILED MODULES

### Module 1 — Simulation Environments
Three Gazebo Harmonic worlds with scripted random object placement:
- **Env A:** Sparse room — few obstacles, open space, uniform lighting.
- **Env B:** Cluttered warehouse — dense obstacles, varied placement, complex shadowing.
- **Env C:** Corridor — narrow passage, objects lining the walls, uneven lighting.

### Module 2 — Synthetic Data Pipeline
- TurtleBot3 Waffle Pi robot drives autonomously through each environment.
- ROS2 camera topic (`/camera/image_raw` or equivalent) → custom image capture node (Python).
- Programmatic bounding box generation: project 3D coordinates from Gazebo's model state API onto the 2D image plane using camera intrinsics.
- Automated train/val split export.
- Target: 500–800 images per environment (1500–2400 total) saved in standard YOLO format.

### Module 3 — Training Pipeline
Three YOLOv8 models trained using Ultralytics API on the HPC GPU cluster:
- **Model A**: Trained exclusively on Env A dataset.
- **Model B**: Trained exclusively on Env B dataset.
- **Model Mixed**: Trained on an equal blend of datasets from Envs A, B, and C.
All training jobs run asynchronously using SLURM batch scripts.

### Module 4 — Benchmark Engine (Research Core)
Cross-test all 3 models across all 3 validation environments to evaluate generalizability:
- Evaluate a 3x3 model-environment testing matrix.
- Key metrics: Mean Average Precision (mAP@50, mAP@50-95), Precision, Recall, Inference Latency (FPS), and visual failure mode analysis.

### Module 5 — Transfer Learning Experiment (Few-Shot Adaptation)
- Take Model B (trained on Env B) and freeze its backbone feature extractor layers.
- Fine-tune it using only 50 images from Env A (simulating low-data domain transfer).
- Compare its performance recovery on Env A against training Model A from scratch and against Model B before adaptation.

### Module 6 — Statistical Decision Layer & Streamlit Dashboard
- Formulate a weighted deployment recommendation score:
  $$score = (w_1 \times mAP) + \left(w_2 \times \frac{1}{\sigma^2_{shift}}\right) + (w_3 \times FPS_{norm})$$
- Develop a Streamlit dashboard visualizing the 3x3 evaluation matrix, domain shift degradation charts, few-shot training curves, and an interactive deployment scoring panel.

---

## 7-WEEK STEP-BY-STEP LEARNING ROADMAP

### Week 1: ROS2 Jazzy, Gazebo Harmonic, & TurtleBot3 Waffle Pi Workspace Setup
*   **Focus & Objectives**: Bring up a functional robotics simulation workspace on the HPC cluster using Ubuntu 24.04 and configure the bridge between ROS2 and Gazebo.
*   **Step-by-Step Execution**:
    1.  Initialize a Colcon workspace at `~/robot_yolo_project/ros2_ws/src`.
    2.  Set up environment variables for ROS2 Jazzy, Gazebo Harmonic, and `TURTLEBOT3_MODEL=waffle_pi`.
    3.  Configure the ROS-Gazebo bridge (`ros_gz_bridge`) to translate simulation clock, robot command velocity, TF transforms, and camera images.
    4.  Verify the setup by spawning a standard TurtleBot3 in a default world and checking the camera topic using `ros2 topic echo /camera/image_raw`.
*   **Skills to Learn**:
    *   **ROS2 Middleware & Nodes**: Understanding topics, publishers, subscribers, workspace sourcing, and compilation.
    *   **Gazebo Harmonic Integration**: Understanding how Ignition/Harmonic Gazebo handles plugins, physics, and bridge mapping files (`.yaml`).
*   **💡 Skill Highlight: Robot-to-Simulator Communication Bridging** — Learning how to establish message-passing interfaces between a physics simulator and robot controller.
*   **Deliverable**: TurtleBot3 successfully spawned in Gazebo, and camera image messages verified via terminal.

### Week 2: Autonomous Obstacle-Avoidance Navigation & Image Capture Node
*   **Focus & Objectives**: Implement a navigation behavior that drives the robot through the space while capturing high-quality camera images.
*   **Step-by-Step Execution**:
    1.  Create a custom Python ROS2 node `obstacle_avoidance_node` in the `robot_drive` package. It subscribes to `/scan` (LiDAR) and publishes to `/cmd_vel` (velocity commands).
    2.  Develop a `camera_capture_node` in the `data_capture` package to listen to the camera image topic and write files to disk.
    3.  Implement spatial or temporal throttling (e.g., save only if the robot has moved >0.5 meters or after 2 seconds) to avoid high dataset correlation (duplicate data).
    4.  Create a master ROS2 launch file to start both nodes together.
*   **Skills to Learn**:
    *   **Lidar-Based Obstacle Avoidance**: Simple reactive control algorithms (e.g., wall-following or random walk with safety stop).
    *   **CV Bridge & OpenCV**: Converting ROS2 `sensor_msgs/Image` to OpenCV `Mat` format in Python and saving PNG files.
    *   **ROS2 Launch System**: Writing Python-based launch scripts that handle parameters and dependencies.
*   **💡 Skill Highlight: Real-Time Sensor Processing** — Subscribing to and parsing raw 2D LiDAR data arrays to drive motor actuators dynamically.
*   **Deliverable**: End-to-end pipeline running: robot driving autonomously and outputting raw frame files to `datasets/raw/`.

### Week 3: Gazebo World Design & Headless HPC Simulation Controls
*   **Focus & Objectives**: Design three distinct test worlds modeling different structural environments and configure them to run efficiently on a headless HPC server.
*   **Step-by-Step Execution**:
    1.  Design World A (Sparse Room), World B (Cluttered Warehouse), and World C (Narrow Corridor) using Gazebo's model library and SDF formats.
    2.  Set up X11 forwarding (`ssh -X`) or use VNC/physical desktop to construct the initial physical layout in the Gazebo GUI, saving them as `.sdf` world files.
    3.  Configure the worlds with random, script-based object placements to ensure dataset diversity.
    4.  Create launch files that run Gazebo Harmonic server-only (headless mode with `-s` flag) to bypass GUI overhead on the HPC.
*   **Skills to Learn**:
    *   **SDF Format Architecture**: Constructing environments, defining collision surfaces, ambient lighting properties, and light sources.
    *   **Headless GPU Rendering**: Running graphical and physics simulations on headless remote servers using cluster engines.
*   **💡 Skill Highlight: Simulation Environment Design & Control** — Designing standardized virtual testbeds that represent varying physical constraints and lighting profiles.
*   **Deliverable**: Three customized `.sdf` files ready to run headless in the pipeline.

### Week 4: Automated Ground-Truth Projection & YOLO Dataset Preparation
*   **Focus & Objectives**: Programmatically annotate the captured images with 2D bounding boxes using simulated ground truth instead of human manual labeling.
*   **Step-by-Step Execution**:
    1.  Access model coordinates and bounding boxes in Gazebo using the simulation's state API (such as topic `/world/<world_name>/dynamic_pose/info`).
    2.  Write a projection script using camera intrinsic parameters to convert 3D coordinates in the camera's frame to 2D pixel coordinates on the image plane.
    3.  Format the projected bounding boxes into normalized YOLO coordinates: `[class_id, x_center, y_center, width, height]`.
    4.  Create a visual validation script to overlay bounding boxes on images to verify projection math.
    5.  Partition datasets into `train` and `val` directories (80/20 split) inside `datasets/env_a`, `datasets/env_b`, and `datasets/env_c`.
*   **Skills to Learn**:
    *   **3D-to-2D Coordinate Projection**: Matrix transformations (translation, rotation, intrinsics calibration projection).
    *   **YOLO Data Label Formats**: Normalizing bounding boxes and structuring folders for training.
*   **💡 Skill Highlight: Synthetic Labeling & Camera Geometry** — Transforming physical 3D world space coordinates to 2D image coordinates using camera projection models.
*   **Deliverable**: Visualized debug images verifying label alignment, and 1500–2400 labeled images structured for YOLO training.

### Week 5: SLURM-Managed YOLOv8 Deep Learning Training on CUDA GPU
*   **Focus & Objectives**: Train the object detection models using high-performance cluster computing infrastructure via job schedulers.
*   **Step-by-Step Execution**:
    1.  Configure a Python virtual environment on the HPC and install PyTorch with CUDA support, along with the `ultralytics` package.
    2.  Write python script templates for Model A, Model B, and Model Mixed training, passing the custom dataset config yaml paths.
    3.  Write SLURM shell batch scripts (`slurm_train.sh`) outlining resource parameters: CPU cores, GPU selection (RTX 5090), VRAM allocation, and log file paths.
    4.  Submit jobs to the cluster queue using `sbatch` and monitor progress using `squeue`.
    5.  Download the final weights (`best.pt`) and analyze performance logs (Loss curves, mAP).
*   **Skills to Learn**:
    *   **SLURM Cluster Computing**: Understanding partition queuing, memory limits, environment isolation, and batch execution scripts.
    *   **ML Model Training Optimization**: Tuning hyperparameters (epochs, learning rate, batch size) for deep neural networks.
*   **💡 Skill Highlight: High-Performance Deep Learning Orchestration** — Managing computational resource scheduling and executing parallel neural network training runs on dedicated GPU hardware.
*   **Deliverable**: Three trained YOLOv8 weight files (`best.pt`) along with their respective SLURM log files.

### Week 6: Cross-Evaluation Benchmarking & Few-Shot Layer Freezing Transfer
*   **Focus & Objectives**: Quantify model domain degradation (domain shift) and evaluate the performance of transfer learning in adapting models to new environments.
*   **Step-by-Step Execution**:
    1.  Develop `run_matrix.py` to evaluate the 3 models on the 3 validation datasets (9 combinations).
    2.  Calculate metrics for each run: Average Precision (mAP@50 and mAP@50:95), precision, recall, and FPS.
    3.  Write a script to load Model B (trained on warehouse data) and freeze its initial convolutional layers (backbone).
    4.  Fine-tune the model on a tiny set of 50 images from Env A (Sparse Room) and measure recovery rate.
    5.  Export all results into structured JSON/CSV files.
*   **Skills to Learn**:
    *   **Domain Shift & Covariate Shift Evaluation**: Understanding generalization boundaries and how variance in data distribution impacts inference.
    *   **Transfer Learning & Layer Freezing**: Modifying optimizer groups in PyTorch/YOLO to update only specific layer weights.
*   **💡 Skill Highlight: Few-Shot Domain Adaptation** — Applying layer-freezing transfer learning techniques to recover object detection performance under domain shift conditions with minimal data.
*   **Deliverable**: Evaluation matrices and fine-tuning performance logs saved in `benchmark/results/`.

### Week 7: Deployment Recommendation Layer & Streamlit Dashboard
*   **Focus & Objectives**: Build an interactive analysis dashboard and implement a decision algorithm to recommend the best model configuration based on user-weighted metrics.
*   **Step-by-Step Execution**:
    1.  Define a deployment score algorithm combining mAP, domain shift variance, and frame rate.
    2.  Develop a Streamlit app (`dashboard/app.py`) displaying comparative bar charts, confusion matrices, and inference latency charts.
    3.  Add interactive slider widgets in the dashboard allowing users to alter the weights ($w_1, w_2, w_3$) to dynamically update model rankings.
    4.  Test and deploy the Streamlit dashboard on the HPC local server port and establish port forwarding or tunnel connection for browser viewing.
*   **Skills to Learn**:
    *   **Multi-Criteria Decision Making (MCDM)**: Translating conflicting engineering parameters (accuracy vs. speed vs. robustness) into a single scoring framework.
    *   **Dashboard Development**: Rapid prototyping of web applications using Streamlit, Plotly, and reactive rendering states.
*   **💡 Skill Highlight: Explainable AI (XAI) & Deployment Analytics** — Packaging raw deep learning metrics into an interactive, user-configurable system dashboard for deployment decision support.
*   **Deliverable**: Streamlit web application running locally and visual presentation of the final portfolio.

---

## CV ONE-LINER

*"Built an autonomous robot perception benchmarking system using ROS2 Jazzy and Gazebo Harmonic, generating synthetic datasets across 3 simulated environments and evaluating YOLOv8 models under domain shift conditions. Demonstrated few-shot domain adaptation by fine-tuning across environments with 50 images. Developed a weighted scoring model for environment-aware deployment decisions, visualized via a Streamlit dashboard."*

---

## SKILLS GAINED

| Skill | Robotics Jobs | AI/ML Jobs |
|---|---|---|
| ROS2 Jazzy | ✅ Core requirement | ➖ Bonus |
| Gazebo Harmonic simulation | ✅ Core requirement | ➖ Bonus |
| Synthetic data generation | ✅ Strong | ✅ Strong |
| YOLOv8 training pipeline | ✅ Good | ✅ Good |
| Domain shift analysis | ✅ Strong | ✅ Strong |
| Transfer learning / fine-tuning | ✅ Good | ✅ Strong |
| SLURM HPC job management | ✅ Good | ✅ Good |
| Streamlit dashboard | ✅ Bonus | ✅ Bonus |

---

## PROJECT FOLDER STRUCTURE (target)

```
robot_yolo_project/
├── envs/                    # Gazebo world files (.sdf)
│   ├── env_a_sparse.sdf
│   ├── env_b_warehouse.sdf
│   └── env_c_corridor.sdf
├── ros2_ws/                 # ROS2 workspace
│   └── src/
│       ├── robot_drive/     # Autonomous drive node
│       └── data_capture/    # Image capture + labeling node
├── datasets/                # YOLO-format datasets
│   ├── env_a/
│   ├── env_b/
│   └── env_c/
├── training/                # YOLOv8 training scripts + SLURM jobs
│   ├── train_model_a.py
│   ├── train_model_b.py
│   ├── train_model_mixed.py
│   └── slurm_train.sh
├── benchmark/               # Cross-evaluation engine
│   ├── run_matrix.py
│   └── results/
├── analysis/                # Charts, domain shift analysis
│   └── plot_results.py
├── dashboard/               # Streamlit app
│   └── app.py
└── SCOPE.md                 # This file
```

---

## AGENT GUIDELINES (Read before every session)

These rules apply when using any AI coding assistant (Claude, Copilot, Cursor, etc.) on this project.

### ✅ Always do
- Always specify Ubuntu 24.04 + ROS2 Jazzy + Gazebo Harmonic when asking for help
- Always mention HPC SSH access (no local display unless X11 forwarding)
- Paste exact error messages — never paraphrase them
- Run one step at a time and verify before moving to the next
- Ask for SLURM job scripts for anything that runs longer than 5 minutes
- Keep ROS2 workspace at `~/robot_yolo_project/ros2_ws/`

### ❌ Never do
- Never ask agent to write full project at once — go module by module
- Never use ROS2 Humble or Gazebo Classic commands — wrong versions
- Never skip verification steps (e.g. don't assume install worked, always run the check command)
- Never run heavy training directly in SSH terminal — always use SLURM
- Never mix this project's code or context with SIGMA-V FYP

### 📋 Per-session checklist
1. State which week and module you are on
2. State what the last step was and whether it succeeded
3. Paste any error output fully
4. Ask for one step at a time

### 🔁 Standard session opener (copy-paste this)
```
Project: YOLO Robot Generalization (CV project, separate from FYP)
OS: Ubuntu 24.04 | ROS2 Jazzy | Gazebo Harmonic | RTX 5090 HPC
Current week: [X] | Current module: [X]
Last completed step: [describe]
Current issue / next step: [describe]
```

---

## IMPORTANT TECHNICAL NOTES

- **ROS2 version:** Jazzy only. Jazzy EOL = May 2029.
- **Gazebo version:** Harmonic only. Classic/Gazebo 11 deprecated on Ubuntu 24.04.
- **TurtleBot3 model:** Waffle Pi (has camera). Not Burger (no camera).
- **CUDA:** 13.1 on RTX 5090 — use `torch.cuda.is_available()` to verify PyTorch sees GPU
- **Headless Gazebo:** Use `gz sim -s` (server only) for data collection without display
- **X11 for GUI:** Use `ssh -X` over ngrok for Gazebo GUI world editing sessions
- **ngrok port:** Changes on restart — retrieve with `curl http://localhost:4040/api/tunnels`

---

*Last updated: Week 0 — Pre-setup*
