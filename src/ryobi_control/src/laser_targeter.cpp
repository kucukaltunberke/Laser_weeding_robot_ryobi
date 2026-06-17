#include "ryobi_control/laser_targeter.h"
#include "ryobi_control/TestServos.h"
#include <cmath>

const double PI = 3.14159265358979323846;

LaserTargeter::LaserTargeter(ros::NodeHandle &nh)
    : nh_(nh), current_state_(TargeterState::IDLE) {
  // Subscribe to weed list (list of Point32 inside a Polygon)
  weed_sub_ =
      nh_.subscribe("/weed_list", 10, &LaserTargeter::weedCallback, this);

  // Publishers for dynamixel laser controllers
  joint_trajectory_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
      "/dynamixel_workbench/joint_trajectory", 10);

  test_servos_srv_ = nh_.advertiseService(
      "/test_servos", &LaserTargeter::testServosCallback, this);

  target_coordinate_srv_ = nh_.advertiseService(
      "/target_coordinate", &LaserTargeter::targetCoordinateCallback, this);

  // New: service to trigger the full detect-and-target pipeline
  start_targeting_srv_ = nh_.advertiseService(
      "/start_targeting", &LaserTargeter::startTargetingCallback, this);

  // New: client to trigger weed detection on demand
  process_image_client_ =
      nh_.serviceClient<std_srvs::Trigger>("/process_latest_image");

  ROS_INFO("Laser Targeter Node Initialized in IDLE state.");
}

void LaserTargeter::weedCallback(const geometry_msgs::Polygon::ConstPtr &msg) {
  // Only update the list of plants when IDLE.
  // While targeting, we ignore new detections so we process the current batch
  // cleanly.
  if (current_state_ == TargeterState::IDLE) {
    latest_weeds_ = *msg;
  }
}

bool LaserTargeter::startTargetingCallback(std_srvs::Trigger::Request &req,
                                           std_srvs::Trigger::Response &res) {
  if (current_state_ == TargeterState::TARGETING) {
    ROS_WARN("Already in TARGETING state. Ignoring call.");
    res.success = false;
    res.message = "Already targeting. Please wait.";
    return true;
  }

  // Stay in IDLE so weedCallback accepts the incoming weed list
  ROS_INFO("Starting targeting pipeline...");

  // 1. Call the weed detector to process the latest camera image
  ROS_INFO("Calling /process_latest_image to detect weeds...");
  std_srvs::Trigger trigger_srv;
  if (!process_image_client_.call(trigger_srv)) {
    ROS_ERROR("Failed to call /process_latest_image service.");
    res.success = false;
    res.message = "Failed to call weed detection service.";
    return true;
  }

  if (!trigger_srv.response.success) {
    ROS_WARN("Weed detection failed: %s", trigger_srv.response.message.c_str());
    res.success = false;
    res.message = "Weed detection failed: " + trigger_srv.response.message;
    return true;
  }

  ROS_INFO("Weed detection successful. Waiting for weed_list update...");

  // 2. Give time for the /weed_list topic to arrive via weedCallback
  //    (state is still IDLE, so weedCallback will accept it)
  ros::Duration(0.5).sleep();
  ros::spinOnce();

  // 3. NOW switch to TARGETING to lock the weed list
  current_state_ = TargeterState::TARGETING;
  ROS_INFO("State changed to TARGETING.");

  // 4. Process the detected weeds
  if (latest_weeds_.points.empty()) {
    ROS_INFO("No weeds detected. Returning to IDLE.");
    current_state_ = TargeterState::IDLE;
    res.success = true;
    res.message = "No weeds detected in the image.";
    return true;
  }

  ROS_INFO("Found %zu weeds. Targeting them one by one.",
           latest_weeds_.points.size());

  // 4. Target each weed with the camera-to-laser offset
  for (size_t i = 0; i < latest_weeds_.points.size(); ++i) {
    auto &pt = latest_weeds_.points[i];
    targetWeedWithOffset(pt.x, pt.y, pt.z, i, latest_weeds_.points.size());
  }

  ROS_INFO("Finished targeting sequence. Returning to IDLE.");
  current_state_ = TargeterState::IDLE;
  res.success = true;
  res.message = "Targeting complete. Processed " +
                std::to_string(latest_weeds_.points.size()) + " weed(s).";
  return true;
}

void LaserTargeter::targetWeedWithOffset(double x_g, double y_g, double z_g,
                                         size_t index, size_t total) {
  // Apply 5cm X offset to account for camera-to-laser distance
  double x_rel = 0.05 + x_g;
  double y_rel = y_g;
  double z = 0.6;

  // Calculate Pan Angle (Rotation around Z)
  double pan_angle = 0.26 + std::atan2(y_rel, x_rel);

  double r = std::sqrt(y_rel * y_rel + x_rel * x_rel);
  double tilt_angle = -0.09 + calculateTiltAngle(r, z);

  // Check limits
  if (pan_angle < -0.5 || pan_angle > 1.2 || tilt_angle < -0.6 ||
      tilt_angle > 0.6) {
    ROS_WARN("Weed %zu/%zu at (%.2f, %.2f, %.2f) -> Pan: %.2f rad, "
             "Tilt: %.2f rad (OUT OF LIMITS, skipping)",
             index + 1, total, x_g, y_g, z_g, pan_angle, tilt_angle);
    return;
  }

  // Publish commands via JointTrajectory
  trajectory_msgs::JointTrajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.joint_names.push_back("pan");
  traj_msg.joint_names.push_back("tilt");

  trajectory_msgs::JointTrajectoryPoint point;
  point.positions.push_back(pan_angle);
  point.positions.push_back(tilt_angle);
  point.time_from_start =
      ros::Duration(0.1); // Short duration for quick reaction

  traj_msg.points.push_back(point);
  joint_trajectory_pub_.publish(traj_msg);

  ROS_INFO("Targeting weed %zu/%zu at (%.2f, %.2f, %.2f) -> Pan: %.2f rad, "
           "Tilt: %.2f rad",
           index + 1, total, x_g, y_g, z_g, pan_angle, tilt_angle);

  // Wait 4 seconds for the laser to stay targeted
  ros::Duration(4.0).sleep();
}

bool LaserTargeter::testServosCallback(
    ryobi_control::TestServos::Request &req,
    ryobi_control::TestServos::Response &res) {
  // Check limits
  if (req.pan < -0.5 || req.pan > 1.2 || req.tilt < -0.6 || req.tilt > 0.6) {
    res.success = false;
    res.message =
        "Requested angles out of limits: Pan [-0.5, 1.2], Tilt [-0.6, 0.6]";
    ROS_WARN(
        "Test command rejected: Pan: %.2f rad, Tilt: %.2f rad (OUT OF LIMITS)",
        req.pan, req.tilt);
    return true;
  }

  ROS_INFO("Executing validation: moving pan and tilt servos to Pan: %.2f rad, "
           "Tilt: %.2f rad",
           req.pan, req.tilt);

  trajectory_msgs::JointTrajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.joint_names.push_back("pan");
  traj_msg.joint_names.push_back("tilt");

  trajectory_msgs::JointTrajectoryPoint point;
  point.positions.push_back(req.pan);
  point.positions.push_back(req.tilt);
  point.time_from_start = ros::Duration(1.0);

  traj_msg.points.push_back(point);
  joint_trajectory_pub_.publish(traj_msg);

  res.success = true;
  res.message = "Test command sent to servos.";
  return true;
}

bool LaserTargeter::targetCoordinateCallback(
    ryobi_control::TargetCoordinate::Request &req,
    ryobi_control::TargetCoordinate::Response &res) {
  double x_g = req.x;
  double y_g = req.y;
  double z_g = req.z;

  // Use req.laser_z, fallback to 0.6 if not provided (0.0)
  double z = (req.laser_z != 0.0) ? req.laser_z : 0.6;

  // Use req.beta, fallback to std::atan(10.0 / 34.0) if not provided (0.0)
  double beta_val = (req.beta != 0.0) ? req.beta : std::atan(10.0 / 34.0);

  // Kinematic translation variables based on distance between camera and laser
  double x_rel = x_g;
  double y_rel = y_g;

  // Calculate Pan Angle (Rotation around Z)
  double pan_angle = 0.28 + std::atan2(y_rel, x_rel);

  double r = std::sqrt(y_rel * y_rel + x_rel * x_rel);
  double tilt_angle = -0.09 + calculateTiltAngle(r, z, beta_val);

  // Check limits
  if (pan_angle < -0.5 || pan_angle > 1.2 || tilt_angle < -0.6 ||
      tilt_angle > 0.6) {
    ROS_WARN("Targeting coordinate at (%.2f, %.2f, %.2f) -> Pan: %.2f rad, "
             "Tilt: %.2f rad (OUT OF LIMITS)",
             x_g, y_g, z_g, pan_angle, tilt_angle);
    res.success = false;
    res.message =
        "Calculated angles out of limits: Pan = " + std::to_string(pan_angle) +
        " (limits: [-0.5, 1.2]), Tilt = " + std::to_string(tilt_angle) +
        " (limits: [-0.6, 0.6])";
    return true;
  }

  ROS_INFO("Targeting coordinate at (%.2f, %.2f, %.2f) with laser_z: %.2f, "
           "beta: %.4f -> Pan: %.2f rad, "
           "Tilt: %.2f rad",
           x_g, y_g, z_g, z, beta_val, pan_angle, tilt_angle);

  trajectory_msgs::JointTrajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.joint_names.push_back("pan");
  traj_msg.joint_names.push_back("tilt");

  trajectory_msgs::JointTrajectoryPoint point;
  point.positions.push_back(pan_angle);
  point.positions.push_back(tilt_angle);
  point.time_from_start = ros::Duration(1.0);

  traj_msg.points.push_back(point);
  joint_trajectory_pub_.publish(traj_msg);

  res.success = true;
  res.message = "Targeting command sent to servos.";
  return true;
}

double LaserTargeter::calculateTiltAngle(double r, double z, double beta_val) {
  double a = 0.055; // Bracket length in meters
  double alpha_beam = PI - beta_val;

  double hyp = std::sqrt(z * z + r * r);
  if (hyp == 0.0) {
    return 0.0;
  }

  // Law of Sines: hyp / sin(alpha_beam) = a / sin(gamma) -> sin(gamma) = (a *
  // sin(alpha_beam)) / hyp
  double sin_gamma = (a * std::sin(alpha_beam)) / hyp;

  // Clamp input of std::asin to protect against precision errors
  if (sin_gamma > 1.0)
    sin_gamma = 1.0;
  if (sin_gamma < -1.0)
    sin_gamma = -1.0;
  double gamma = std::asin(sin_gamma);

  // Interior angle sum: beta = beta_val - gamma
  double beta = beta_val - gamma;

  // Final servo control angle
  double theta = (PI / 2.0) - beta - std::atan2(r, z);
  return theta;
}
