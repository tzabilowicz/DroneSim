# controller.py
"""
Basic flight controller for a drone model.
Controls the drone to takeoff and hover at the preset altitude.
"""

import numpy as np

class Controller:
    def __init__(self, target_altitude = 10.0):
        """ Initialize the controller parameters. """

        self.target_altitude = target_altitude

        # Hover thrust from x2.xml keyframe
        self.hover = 3.2495625

        # PD gains
        self.kp = 1.2
        self.kd = 0.8

        # Small attitude stabilization
        self.roll_gain = 0.15
        self.pitch_gain = 0.15


    def update(self, model, data):
        """ Update the drone model. """

        # Current altitude
        z = data.qpos[2]

        # Vertical velocity
        vz = data.qvel[2]

        altitude_error = self.target_altitude - z

        # Compute the required thrust
        thrust = self.hover + self.kp * altitude_error - self.kd * vz
        thrust = np.clip(thrust, 0.0, 13.0)

        quat = data.qpos[3:7]

        # quaternion = [w,x,y,z]
        w, x, y, zq = quat

        roll = np.arctan2(
            2 * (w * x + y * zq),
            1 - 2 * (x * x + y * y),
        )

        pitch = np.arcsin(
            np.clip(
                2 * (w * y - zq * x),
                -1,
                1,
            )
        )

        roll_corr = self.roll_gain * roll
        pitch_corr = self.pitch_gain * pitch

        # Compute the mixed motor control thrusts
        ctrl = np.array([
            thrust + roll_corr + pitch_corr,
            thrust - roll_corr + pitch_corr,
            thrust - roll_corr - pitch_corr,
            thrust + roll_corr - pitch_corr,
        ])

        ctrl = np.clip(ctrl, 0, 13)

        return ctrl