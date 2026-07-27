# main.py
"""
Environment testing script. Loads the MuJoCo scene, makes a drone takeoff and hover,
randomly renders a target in the scene.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

from controller.controller import Controller
from utils.utils import randomize_target

# Scene path
MJCF_PATH = r"assets\mujoco_menagerie\skydio_x2\scene.xml"


def simulate_drone():
    # Load models
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Set the target position
    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target"
    )

    model.body_pos[target_body_id] = randomize_target(50, 2)

    # Configurable physics
    model.opt.gravity[:] = [0.0, 0.0, -9.81] # Gravity
    wind_velocity = np.array([0.0, 0.0, 0.0])
    model.opt.wind[:] = wind_velocity

    # Air density
    model.opt.density = 1.225

    # Sensor noise
    sensor_noise_config = {
        'accelerometer': 0.1,  # m/s^2
        'gyro': 0.01,          # rad/s
        'magnetometer': 0.05
    }

    def get_sensor_slice(sensor_name):
        """ Get sensor slices from 1D sensor array. """

        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id != -1:
            adr = model.sensor_adr[sensor_id]
            dim = model.sensor_dim[sensor_id]
            return adr, dim
        return None, None

    print("Starting simulation... (Close the viewer window to exit)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        data.qpos[2] = 0.0

        controller = Controller(target_altitude=5.0)

        while viewer.is_running():
            step_start = time.time()

            # Compute motor commands
            data.ctrl[:] = controller.update(model, data)

            # Advance physics
            mujoco.mj_step(model, data)

            measured_sensordata = data.sensordata.copy()

            for sensor_name, std_dev in sensor_noise_config.items():
                adr, dim = get_sensor_slice(sensor_name)

                if adr is not None:
                    noise = np.random.normal(0, std_dev, size=dim)
                    measured_sensordata[adr:adr + dim] += noise

            viewer.sync()

            time_until_next_step = (
                model.opt.timestep - (time.time() - step_start)
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    simulate_drone()