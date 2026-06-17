#ifndef LASER_TARGETER_H
#define LASER_TARGETER_H

#include <ros/ros.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Point32.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <std_srvs/Trigger.h>
#include <ryobi_control/TestServos.h>
#include <ryobi_control/TargetCoordinate.h>
#include <vector>

enum class TargeterState {
    IDLE,
    TARGETING
};

class LaserTargeter {
private:
    ros::NodeHandle nh_;
    ros::Subscriber weed_sub_;
    ros::Publisher joint_trajectory_pub_;
    ros::ServiceServer test_servos_srv_;
    ros::ServiceServer target_coordinate_srv_;
    ros::ServiceServer start_targeting_srv_;
    ros::ServiceClient process_image_client_;

    geometry_msgs::Polygon latest_weeds_;
    TargeterState current_state_;

    void weedCallback(const geometry_msgs::Polygon::ConstPtr& msg);
    bool testServosCallback(ryobi_control::TestServos::Request& req, ryobi_control::TestServos::Response& res);
    bool targetCoordinateCallback(ryobi_control::TargetCoordinate::Request& req, ryobi_control::TargetCoordinate::Response& res);
    bool startTargetingCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
    void targetWeedWithOffset(double x_g, double y_g, double z_g, size_t index, size_t total);
    double calculateTiltAngle(double r, double z, double beta_val = 0.2855799332);

public:
    LaserTargeter(ros::NodeHandle& nh);
};

#endif // LASER_TARGETER_H

