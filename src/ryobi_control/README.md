# ryobi_control

The `ryobi_control` package operates as the primary node structure for issuing mechanical logic to the robot's behavior suite. It acts as the execution layer that links detection coordinate offsets extrapolated from the image processing package, converting them into physical hardware manipulation (laser orienting) utilizing C++ handlers.

## How to Run

1. Ensure you have sourced your ROS workspace:
   ```bash
   source devel/setup.bash
   ```

2. Start the primary control node:
   ```bash
   rosrun ryobi_control laser_targeter
   ```

### Demonstration
https://github.com/kucukaltunberke/laser_weeding_robot/raw/main/src/laser_test.mp4
