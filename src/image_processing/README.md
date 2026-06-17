# image_processing

This package handles the computer vision and spatial coordinate solving components of the weed targeting robot. It executes deep learning inference on live camera feeds and translates 2D pixel coordinates into physical 3D ground coordinates.

## 📋 Core Modules

*   **`weed_detector.py`**: A ROS Python node that subscribes to synchronized RGB (`/zedm/zed_node/left/image_rect_color`) and depth (`/zedm/zed_node/depth/depth_registered`) streams from the **ZED Mini** stereo camera. It executes native GPU-accelerated **TensorRT** inference using a custom YOLOv8 model (`new_weed_detector.engine`).
*   **`weed_coordinate_solver.py`**: A helper module that translates the detected bounding box center $(u, v)$ and true depth $z$ into physical coordinates $(X_g, Y_g, Z_g)$ relative to the ground beneath the camera, accounting for camera pitch ($37.7425^\circ$) and height ($0.35472\text{ m}$). It features a geometric ground-plane intersection fallback if the depth sensor gets blinded.

## 🚀 How to Run

1.  Ensure you have sourced your ROS workspace:
    ```bash
    source devel/setup.bash
    ```

2.  Ensure the ZED Mini camera wrapper is running:
    ```bash
    roslaunch zed_wrapper zedm.launch
    ```

3.  Launch the YOLOv8 weed detection node:
    ```bash
    rosrun image_processing weed_detector.py
    ```

4.  To visualize the annotated results in real-time, run `rqt_image_view`:
    ```bash
    rosrun rqt_image_view rqt_image_view
    ```
    Then, select the topic `/yolo/annotated_image/compressed` from the dropdown menu to see the live weed detections with confidence values and 3D metric coordinate annotations.
