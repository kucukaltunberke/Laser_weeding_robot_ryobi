# ryobi_control

The `ryobi_control` package acts as the execution layer that subscribes to weed detection coordinate outputs and performs the inverse kinematics and control commands for the physical pointer laser targeting system.

## 📋 Core Components

*   **`src/laser_targeter.cpp`**: A C++ ROS node that subscribes to the `/weed_list` topic. It applies a physical $5\text{cm}$ offset along the X-axis (for the camera-to-laser distance), calculates the target pan angle using `atan2`, and computes the tilt angle using the **Law of Sines** based on a physical bracket length of $0.055\text{ m}$ and mounting angle $\beta \approx 0.2856\text{ rad}$. The calculated angles are published to the Dynamixel controller.
*   **`config/dynamixel_config.yaml`**: Configuration parameters for the Dynamixel actuators defining:
    *   **Pan Servo (ID 1)**: Return Delay Time and Velocity Profile.
    *   **Tilt Servo (ID 2)**: Return Delay Time and Velocity Profile.
*   **`launch/dynamixel.launch`**: Wrapper launch file that interfaces with `dynamixel_workbench_controllers` to initialize serial communications via `/dev/ttyUSB0` at 57600 baud.
*   **`launch/ryobi_bringup.launch`**: Master bringup file that launches the ZED camera, YOLOv8 detector, Dynamixel drivers, and the laser targeter node concurrently.

## 🚀 How to Run

1.  Ensure you have sourced your ROS workspace:
    ```bash
    source devel/setup.bash
    ```

2.  Start the entire system using the main bringup:
    ```bash
    roslaunch ryobi_control ryobi_bringup.launch
    ```

3.  Alternatively, launch the Dynamixel driver and targeter node individually for debugging:
    ```bash
    roslaunch ryobi_control dynamixel.launch
    rosrun ryobi_control laser_targeter
    ```
