#include <ros/ros.h>
#include "ryobi_control/laser_targeter.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_targeter");
    ros::NodeHandle nh;

    LaserTargeter targeter(nh);

    ros::AsyncSpinner spinner(2);
    spinner.start();

    ROS_INFO("Laser Targeter ready. Call /start_targeting to begin.");
    ros::waitForShutdown();
    
    return 0;
}

