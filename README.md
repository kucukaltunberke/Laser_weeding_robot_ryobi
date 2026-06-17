# 🤖 Autonomous Weed Elimination System (Ryobi Control)

An advanced, high-performance ROS (Robot Operating System) robotic vision and laser-targeting pipeline deployed on the **NVIDIA Jetson Nano** platform. This repository integrates real-time object detection via **YOLOv8** accelerated with **TensorRT**, spatial coordinate solving using **ZED Mini** stereo depth camera intrinsics, and precise physical target acquisition using a custom **Dynamixel** pan-tilt servo mechanism guiding a targeting laser.

---

## 📐 System Architecture

```mermaid
graph TD
    subgraph Hardware Peripherals
        ZED[Stereolabs ZED Mini Camera]
        Laser[Class 4 Laser Targeter]
        Servos[Dynamixel Servos: Pan & Tilt]
    end

    subgraph Computing Platform (NVIDIA Jetson Nano)
        subgraph ROS Core Workspace
            GUI[PyQt5 GUI Control Panel]
            Detector[Weed Detector Node: Python]
            Solver[Coordinate Solver Module: Python]
            Targeter[Laser Targeter Node: C++]
            DxlDriver[Dynamixel Workbench Controllers]
        end
        TRT[TensorRT GPU Engine]
    end

    %% Data Flow
    ZED -->|RGB & Depth Streams| Detector
    Detector -->|Raw Images| TRT
    TRT -->|Bounding Boxes / BBox Centers| Detector
    Detector -->|Pixel u, v & Depth z| Solver
    Solver -->|Calculated 3D Metric Coordinates| Detector
    Detector -->|/weed_list geometry_msgs/Polygon| Targeter
    Targeter -->|Inverse Kinematics / Law of Sines| Targeter
    Targeter -->|/joint_trajectory| DxlDriver
    DxlDriver -->|Physical Position Commands| Servos
    GUI -->|Process Subprocesses / Services| ROS Core Workspace
```

---

## 💻 Hardware Specifications & Configuration

This project is fully tailored to run on the physical boundaries and hardware limits of the **Ryobi Weed Elimination Robot**:

| Hardware Component | Model / Spec | Role in System |
| :--- | :--- | :--- |
| **Computing Unit** | NVIDIA Jetson Nano (4GB RAM) | Runs ROS Master, GPU TensorRT Inference, Inverse Kinematics, and GUI control. |
| **Vision Sensor** | Stereolabs ZED Mini | Captures stereoscopic RGB and Depth streams (registered to the left camera frame). |
| **Pan-Tilt Actuators** | Dynamixel Servos (MX-series) | ID 1 (Pan) & ID 2 (Tilt). Controls angular targeting with high torque/resolution. |
| **End Effector** | Class 4 Laser System | Emits a high-power beam to destroy target weeds at physical locations. |
| **Communication Interface** | USB-to-RS485 (U2D2) | Translates half-duplex TTL/RS485 commands from Jetson to Dynamixel bus. |

---

## 🧠 Software Architecture & Algorithmic Pipeline

The system runs on **ROS Melodic** (Ubuntu 18.04) and is structured into two main packages: `image_processing` (Python-based) and `ryobi_control` (C++ based).

### 1. Vision & Object Detection (`weed_detector.py`)
- **Acceleration**: A custom YOLOv8 model is compiled into a platform-specific **TensorRT Engine** (`new_weed_detector.engine`). Native GPU execution is handled via **PyCUDA** and the **TensorRT C++ API wrappers in Python**, bypassing PyTorch runtime overhead to run in real-time under memory constraints (~4GB RAM Jetson Nano limit).
- **Preprocessing**: Input frames (1080p resolution) are processed using *letterbox resizing* to preserve aspect ratios before tensor feeding.
- **Service-Triggered (On-Demand)**: To optimize CPU/GPU cycles, inference is triggered on-demand via the `/process_latest_image` service, preventing frame buffering latency.

### 2. Spatial 3D Coordinate Solver (`weed_coordinate_solver.py`)
Translates $2\text{D}$ pixel coordinates $(u, v)$ from the camera feed into $3\text{D}$ coordinates $(X_g, Y_g, Z_g)$ relative to the ground origin directly beneath the camera.

#### Mathematical Coordinate Transformation Pipeline:
1. **Optical Camera Frame Projection** (Z forward, X right, Y down):
   $$x_c = \frac{z_{depth} \cdot (u - c_x)}{f_x}$$
   $$y_c = \frac{z_{depth} \cdot (v - c_y)}{f_y}$$
   $$z_c = z_{depth}$$
   *Where $f_x, f_y$ are focal lengths and $c_x, c_y$ represent the optical center (principal point).*

2. **Standard Camera Coordinate Frame** (X forward, Y left, Z up):
   $$x_c' = z_c, \quad y_c' = -x_c, \quad z_c' = -y_c$$

3. **Pitch Angle Rotation**:
   The camera is physically mounted with a downward tilt ($\theta_{pitch} = 37.7425^\circ$). To align coordinates with the robot body, we apply a rotation matrix around the lateral (Y) axis:
   $$x_{rot} = x_c' \cos(\theta_{pitch}) + z_c' \sin(\theta_{pitch})$$
   $$y_{rot} = y_c'$$
   $$z_{rot} = -x_c' \sin(\theta_{pitch}) + z_c' \cos(\theta_{pitch})$$

4. **Ground Translation**:
   $$X_g = x_{rot}$$
   $$Y_g = y_{rot}$$
   $$Z_g = z_{rot} + h_{camera}$$
   *Where $h_{camera} = 0.35472\text{ m}$ is the camera height calculated from the physical chassis limits.*

5. **Blinding Fallback (Geometric Ground Plane Intersection)**:
   If depth sensors get blinded (producing `NaN` or zero depth values), the solver projects a mathematical ray through the camera pitch onto the ground plane ($Z_g = 0$):
   $$z_{depth} = \frac{h_{camera}}{\sin(\theta_{pitch}) + \frac{(v - c_y)}{f_y} \cos(\theta_{pitch})}$$

---

### 3. Laser Kinematics & Aiming (`laser_targeter.cpp`)
Once the target coordinates $(x_g, y_g, z_g)$ are received on `/weed_list`, the targeting node performs kinematic translations to aim the Dynamixel servos.

- **Translation Offset**: A physical $5\text{cm}$ offset along the $X$-axis accounts for the camera-to-laser distance:
  $$x_{rel} = x_g + 0.05$$
  $$y_{rel} = y_g$$
- **Pan Angle ($\theta_{pan}$)**:
  $$\theta_{pan} = 0.26 + \text{atan2}(y_{rel}, x_{rel})$$
- **Tilt Angle ($\theta_{tilt}$)**:
  Using a physical bracket of length $a = 0.055\text{ m}$ and mounting angle $\beta$ (default $\approx 0.2856\text{ rad}$), the tilt angle is resolved using the **Law of Sines**:
  $$\text{hyp} = \sqrt{z_{laser}^2 + r^2} \quad \text{where } r = \sqrt{x_{rel}^2 + y_{rel}^2}$$
  $$\sin(\gamma) = \frac{a \cdot \sin(\pi - \beta)}{\text{hyp}}$$
  $$\theta_{tilt} = -0.09 + \left( \frac{\pi}{2} - (\beta - \gamma) - \text{atan2}(r, z_{laser}) \right)$$
  *(Angles are strictly software-bounded to: Pan $[-0.5, 1.2]\text{ rad}$, Tilt $[-0.6, 0.6]\text{ rad}$).*
- **Targeting Execution**: The laser aims at each coordinate and stays locked for $4.0\text{ seconds}$ to simulate thermal elimination.

---

### 4. Interactive PyQt5 Control Panel (`gui_panel.py`)

A graphical interface that manages ROS subprocess lifecycles, parses real-time logs, and implements Human-Computer Interaction (HCI) safety rules:
- **Sequential Startup**: Encourages starting main operations and sensors in order.
- **Log Simplifier**: Filters out verbose, repetitive ROS logs to display only key events (e.g., detection coordinates, targeting updates, warnings) in color-coded lines.
- **Safety Lock (Error Prevention)**: Arming the laser requires confirmation through a modal warning dialog. When armed, the UI changes layout colors and allows disarming via a prominent stop button.

---

## 🚀 Deployment & Execution Guide

### 1. Build the Workspace
Clean, configure, and compile the workspace on the Jetson Nano:
```bash
cd ~/catkin_ws
catkin_make -DCATKIN_WHITELIST_PACKAGES="ryobi_control;image_processing"
source devel/setup.bash
```

### 2. Launching the Whole System
To spin up all nodes (ZED Mini camera wrapper, TensorRT detector, Dynamixel drivers, laser targeting solver):
```bash
roslaunch ryobi_control ryobi_bringup.launch
```

### 3. Launching Components Individually
For debug mode, open separate terminals:
* **ZED Camera Wrapper**:
  ```bash
  roslaunch zed_wrapper zedm.launch
  ```
* **Weed Detector Node**:
  ```bash
  rosrun image_processing weed_detector.py
  ```
* **Dynamixel Servo Controller**:
  ```bash
  roslaunch ryobi_control dynamixel.launch
  ```
* **Laser Targeter Node**:
  ```bash
  rosrun ryobi_control laser_targeter
  ```

### 4. Running the Local GUI
Run the PyQt5 panel:
```bash
python3 src/gui_panel.py
```

---

## 📡 ROS API Reference

### Published Topics
* `/weed_list` (`geometry_msgs/Polygon`): Metric coordinate array of detected weeds.
* `/yolo/annotated_image/compressed` (`sensor_msgs/CompressedImage`): Compressed annotated JPEG stream with real-time center markers and depth values.
* `/dynamixel_workbench/joint_trajectory` (`trajectory_msgs/JointTrajectory`): Direct commands to the pan/tilt Dynamixel workbench driver.

### ROS Services
* `/start_targeting` (`std_srvs/Trigger`): Runs the full vision-to-aim pipeline.
* `/process_latest_image` (`std_srvs/Trigger`): Captures one frame and performs weed detection.
* `/target_coordinate` (`ryobi_control/TargetCoordinate`): Manually commands the laser targeting node to calculate angles and aim at coordinate $(x, y, z)$.
* `/test_servos` (`ryobi_control/TestServos`): Forces pan and tilt servos to exact raw angles for validation.

---

## ⚠️ Safety Warning
> [!WARNING]
> This system commands a Class 4 laser. Class 4 lasers can cause immediate eye damage and skin burns. Always ensure:
> 1. All operators wear appropriate laser safety goggles certified for the laser wavelength.
> 2. The laser environment is clear of reflective surfaces and bystanders.
> 3. The manual disarm command in `gui_panel.py` is easily accessible during execution.
