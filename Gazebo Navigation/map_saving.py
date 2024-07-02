#!/usr/bin/env python3

import os
import subprocess
import time
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt

def launch_gazebo():
    return subprocess.Popen(["roslaunch", "turtlebot3_gazebo", "turtlebot3_house.launch"])

def launch_slam():
    return subprocess.Popen(["roslaunch", "turtlebot3_slam", "turtlebot3_slam.launch"])

def launch_teleop():
    return subprocess.Popen(["roslaunch", "turtlebot3_teleop", "turtlebot3_teleop_key.launch"])

def save_map_as_image(map_data, map_directory, map_name):
    # Create the directory if it doesn't exist
    os.makedirs(map_directory, exist_ok=True)

    # Convert map data to a numpy array
    width = map_data.info.width
    height = map_data.info.height
    resolution = map_data.info.resolution
    map_array = np.array(map_data.data).reshape((height, width))

    # Replace unknown values (-1) with 0.5 for visualization purposes
    map_array = np.where(map_array == -1, 0.5, map_array)

    # Plot the map and save it as an image
    plt.imshow(map_array, cmap='gray', origin='lower')
    plt.title('Occupancy Grid Map')
    map_path = os.path.join(map_directory, f"{map_name}.png")
    plt.savefig(map_path)
    plt.close()
    print(f"Map saved to: {map_path}")

def visualize_map(map_data):
    width = map_data.info.width
    height = map_data.info.height
    resolution = map_data.info.resolution

    # Convert map data to a numpy array
    map_array = np.array(map_data.data).reshape((height, width))

    # Replace unknown values (-1) with 0.5 for visualization purposes
    map_array = np.where(map_array == -1, 0.5, map_array)
    
    # Plot the map
    plt.imshow(map_array, cmap='gray', origin='lower')
    plt.title('Occupancy Grid Map')
    plt.show()

def map_callback(data):
    global map_data
    map_data = data

def get_map():
    rospy.init_node('map_listener', anonymous=True)
    rospy.Subscriber("/map", OccupancyGrid, map_callback)
    rospy.spin()

if __name__ == "__main__":
    # Define the directory and name for saving the map
    map_directory = os.path.expanduser('~/Desktop')
    map_name = "house_plot"

    # Start roscore
    roscore_process = subprocess.Popen(["roscore"])
    time.sleep(5)  # Allow some time for roscore to start

    # Launch Gazebo
    gazebo_process = launch_gazebo()
    time.sleep(5)  # Allow some time for Gazebo to start

    # Launch SLAM
    slam_process = launch_slam()
    time.sleep(5)  # Allow some time for SLAM to start

    # Launch teleoperation
    teleop_process = launch_teleop()

    try:
        # Keep the script running to allow teleoperation
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # When the user interrupts (Ctrl+C), save the map and clean up
        rospy.init_node('map_listener', anonymous=True)
        map_data = rospy.wait_for_message("/map", OccupancyGrid)
        save_map_as_image(map_data, map_directory, map_name)

        gazebo_process.terminate()
        slam_process.terminate()
        teleop_process.terminate()
        roscore_process.terminate()

    # Initialize ROS node to get the map
    rospy.init_node('map_listener', anonymous=True)
    map_data = rospy.wait_for_message("/map", OccupancyGrid)
    
    visualize_map(map_data)

