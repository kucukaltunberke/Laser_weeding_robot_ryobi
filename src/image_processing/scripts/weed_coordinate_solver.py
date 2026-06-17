#!/usr/bin/env python3.10
import math
import numpy as np

# ==============================================================================
# PARAMETERS FOR 3D COORDINATE SOLVER
# Modify these parameters when transferring to the real robot!
# ==============================================================================

# Physical camera placement
CAMERA_HEIGHT_METERS = 0.35472    # Height of the camera lens from the ground (Calculated from URDF axle limits)
CAMERA_PITCH_DEG = 37.7425        # Downward tilt of the camera (0 = looking straight ahead)

# Camera Intrinsics
# If these change on the real robot, update them here or subscribe to camera_info
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAMERA_FOV_DEG = 80.0             # Horizontal Field of View in degrees

# ==============================================================================

class WeedCoordinateSolver:
    def __init__(self):
        self.camera_height = CAMERA_HEIGHT_METERS
        self.camera_angle_rad = math.radians(CAMERA_PITCH_DEG)
        
        self.fov_h_rad = math.radians(CAMERA_FOV_DEG)
        
        # Pinhole Camera Model
        self.cx = IMAGE_WIDTH / 2.0
        self.cy = IMAGE_HEIGHT / 2.0
        
        # Focal length (assuming square pixels fx = fy)
        self.fx = IMAGE_WIDTH / (2.0 * math.tan(self.fov_h_rad / 2.0))
        self.fy = self.fx
        
        # Pre-calculated trig constants for efficiency
        self.cos_alpha = math.cos(self.camera_angle_rad)
        self.sin_alpha = math.sin(self.camera_angle_rad)

    def get_3d_coordinate(self, u, v, depth_z):
        """
        Takes the (u, v) pixel and true Z depth from the depth camera.
        Returns the (X_g, Y_g, Z_g) physical coordinates relative to the 
        ground directly below the camera.
        
        X_g: Forward distance
        Y_g: Lateral distance (Left is positive)
        Z_g: Height from the ground
        """
        if math.isnan(depth_z) or math.isinf(depth_z) or depth_z <= 0.01:
            # Geometric Ground Intersection Fallback
            # Bypasses sensor blinding by shooting a ray through the pixel straight onto the mathematical ground plane.
            denominator = self.sin_alpha + ((v - self.cy) / self.fy) * self.cos_alpha
            if denominator > 0.0001:  # Prevent divide by zero (horizon pixels)
                depth_z = self.camera_height / denominator
            else:
                return None
            
        # 1. Optical frame coordinates (Z forward, X right, Y down)
        x_c = depth_z * (u - self.cx) / self.fx
        y_c = depth_z * (v - self.cy) / self.fy
        z_c = depth_z
        
        # 2. Standard camera frame (X forward, Y left, Z up)
        x_c_prime = z_c
        y_c_prime = -x_c
        z_c_prime = -y_c
        
        # 3. Rotate by pitch angle (around Y axis)
        x_c_rot = x_c_prime * self.cos_alpha + z_c_prime * self.sin_alpha
        y_c_rot = y_c_prime
        z_c_rot = -x_c_prime * self.sin_alpha + z_c_prime * self.cos_alpha
        
        # 4. Translate by camera height into ground frame
        x_g = x_c_rot
        y_g = y_c_rot
        z_g = z_c_rot + self.camera_height
        
        return (x_g, y_g, z_g)
