import os

# Set the relative location for cloning XML assets
os.environ["ROBOT_DESCRIPTIONS_CACHE"] = r".\assets"

# Fetches the assets
from robot_descriptions import skydio_x2_mj_description