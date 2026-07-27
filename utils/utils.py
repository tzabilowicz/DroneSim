# utils.py

import numpy as np


def randomize_target(width, min_height):
    """ Randomizes the target's (x,y,z) position in meters. """

    # Get the bounds of the square environment.
    bound = width // 2

    x = np.random.uniform(-bound, bound)
    y = np.random.uniform(-bound, bound)
    z = np.random.uniform(min_height, bound)

    return np.array([x, y, z])