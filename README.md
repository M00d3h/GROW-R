# GROW-R: Guided Rover for Watering with Reinforcement Learning 🌿🤖

![Project Demo](demo.gif)
*(Note: To view the demo, ensure a `demo.gif` is present in the root directory, or see `episode.mp4`)*

## 📖 Overview

**GROW-R** is an autonomous reinforcement learning agent designed to navigate complex, procedurally generated environments to perform agricultural tasks. The agent, a rover, is trained to explore a 25x25 grid world, avoid obstacles, and efficiently water "thirsty" plants while managing resource constraints.

This project simulates real-world challenges found in **automated agriculture**, **warehouse robotics**, and **solar panel maintenance**, where agents must operate in dynamic environments with competing objectives (exploration vs. task execution).

Developed as a capstone for a Reinforcement Learning course, this project implements and compares three major RL architectures: **DQN**, **A2C**, and **RecurrentPPO**.

---

## 📂 Repository Structure

```text
RL-Env/
├── assets/             # 🖼️ Environment textures, 3D models, and media
├── models/             # 🤖 Trained RL model weights (.zip)
├── results/            # 📈 Training logs and performance graphs
├── gradio-app/         # 🖥️ Source code for the interactive Gradio interface
├── plantos_env.py      # 🌍 Core Gymnasium environment logic
├── plantos_utils.py    # 🛠️ Helper functions and environment constants
├── example_usage.py    # 🚀 CLI script for running pre-trained agents
├── requirements.txt    # 📦 Python dependencies
└── README.md           # 📜 Project overview and documentation
```

---

## 🌍 Environment Specification

The **PlantOS** environment is a gymnasium-compliant custom grid world designed to test the agent's ability to balance exploration and specific task execution.

### 🗺️ The Grid World
*   **Dimensions:** 25x25 Grid (procedurally generated each episode).
*   **Entities:**
    *   🤖 **Rover:** The agent controlled by the RL algorithm.
    *   🌵 **Thirsty Plants:** Targets that need watering (orange).
    *   🌿 **Hydrated Plants:** Plants that do not need attention (green).
    *   🪨 **Obstacles:** Rocks or walls that block movement (gray).

### 🎮 Action Space (Discrete: 5)
The agent can perform one of five actions at each time step:
0.  **Move North** ⬆️
1.  **Move East** ➡️
2.  **Move South** ⬇️
3.  **Move West** ⬅️
4.  **Water** 💧 (Attempts to water the plant at the current location)

### 👁️ Observation Space (Box: 107)
The agent receives a complex, multi-modal observation vector of size 107, designed to mimic realistic sensor data rather than giving global knowledge.

| Component | Size | Description |
| :--- | :--- | :--- |
| **LIDAR Sensors** | 80 | **16 Directions** around the rover. Each ray returns 5 values: <br>• **Distance** (normalized)<br>• **Entity Type** (One-hot: Empty, Obstacle, Hydrated, Thirsty) |
| **Position** | 2 | Normalized (X, Y) coordinates of the rover. |
| **Local Visit Map** | 25 | A **5x5 grid** centered on the rover. Tracks how many times nearby cells have been visited to discourage loitering and encourage exploration. |

### 🎁 Reward Function
The reward structure is carefully tuned to shape the desired behavior (DQN Config):

*   **+20** | **Goal:** Successfully watering a thirsty plant.
*   **+10** | **Exploration:** Visiting a new cell for the first time.
*   **+50** | **Completion:** Bonus for exploring 100% of the map.
*   **-0.1** | **Time Penalty:** Small penalty per step to encourage efficiency.
*   **-1.0** | **Loitering:** Penalty for revisiting an already explored cell.
*   **-5.0** | **Collision/Invalid:** Hitting a wall or obstacle.
*   **-5.0** | **Waste:** Watering empty ground.
*   **-10.0**| **Mistake:** Watering an already hydrated plant.

---

## 🚀 Key Features

*   **Custom Gymnasium Environment ('PlantOS'):** A procedurally generated 2D grid world with random placement of plants (thirsty/hydrated) and obstacles.
*   **Advanced State Representation:**
    *   **16-Channel LIDAR:** Simulates realistic proximity sensors.
    *   **Local Visit Map:** A 5x5 memory grid to encourage exploration and prevent loitering.
    *   **107-Dimensional Observation Vector.**
*   **Curriculum Learning:** Implements a custom wrapper that progressively increases map difficulty as the agent masters simpler layouts.
*   **Multi-Model Analysis:** Comparative study of Off-Policy (DQN) vs. On-Policy (A2C, RecurrentPPO) algorithms.
*   **Dual Rendering Engine:**
    *   **2D Visualization:** Fast, top-down view using **Pygame** for training and debugging.
    *   **3D Visualization:** immersive "Follow-Camera" view using the **Ursina Engine**.
*   **Interactive UI:** A **Gradio** interface for easy model selection and parameter tuning.

---

## 🧠 The Challenge

The agent must balance multiple objectives to maximize its reward:
1.  **Navigation:** Efficiently move through the grid.
2.  **Obstacle Avoidance:** Detect and navigate around walls and rocks.
3.  **Task Execution:** Identify "thirsty" plants and execute the specific `water` action.
4.  **Resource Management:** Minimize steps and avoid watering empty ground or already hydrated plants.

**State Space:** 107 values (LIDAR + Position + Visit Map).
**Action Space:** 5 Discrete Actions (North, East, South, West, Water).

---

## 📊 Results & Architecture

We trained and evaluated three distinct architectures. Contrary to our initial hypothesis favoring memory-based models (LSTMs), **DQN (Deep Q-Network)** achieved the superior performance.

| Model | Avg. Reward | Exploration % | Success Rate |
|-------|-------------|---------------|--------------|
| **DQN** | **5339.22** | **97.11%** | **100%** |
| RecurrentPPO | 4339.39 | 87.71% | High |
| A2C | 3518.69 | 86.36% | Moderate |

**Why DQN Won:**
Although "memoryless", the inclusion of the **Local Visit Map** in the observation space provided an explicit short-term memory that the stable, off-policy nature of DQN leveraged effectively. This highlights the critical importance of *State Representation* over purely algorithmic complexity.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/M00d3h/GROW-R.git
    cd RL-Env
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If using RecurrentPPO, ensure `sb3-contrib` is installed.*

---

## 🎮 Usage

You can run the pre-trained agents using the provided usage script.

**To run the best performing model (DQN):**
```bash
python example_usage.py models/dqn_model_weights.zip --model-type dqn
```

**To use the Gradio UI:**
```bash
python gradio-app/gradioUI.py
```

---