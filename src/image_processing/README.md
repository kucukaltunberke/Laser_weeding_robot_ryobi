# image_processing


This package is dedicated to running image processing algorithms for agricultural analysis. It utilizes a trained YOLOv8 model subscribing to raw camera images emitted from Gazebo to identify models such as weeds vs. crop variants. 

## How to Run

1. Ensure you have sourced your ROS workspace:
   ```bash
   source devel/setup.bash
   ```

2. Launch the YOLOv8 weed detection node:
   ```bash
   rosrun image_processing weed_detector.py
   ```

3. To visualize the annotated results in real-time, open `rqt_image_view`:
   ```bash
   rosrun rqt_image_view rqt_image_view
   ```
   Then, select the topic `/yolo/annotated_image` from the dropdown menu to see the live weed detections.

### Demonstration
https://github.com/kucukaltunberke/laser_weeding_robot/raw/main/src/image_processing_test.mp4

> [!NOTE]
> **Future Improvements**: Further improvement on the `image_processing` package is going to be focused heavily on **filtering**. Developing logical stability sweeps over frame-to-frame bounding boxes will guarantee smoother inputs for downstream kinematics.

