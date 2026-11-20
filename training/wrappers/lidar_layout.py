# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Six goal-relative LiDAR rays - the one place that defines where every ray points.

The simulator, the reward's proximity shaping and the expert all import from
here, so they cannot disagree about which reading means "front".

Why goal-relative. The drone never yaws: VelocityAviary holds the current yaw,
and a measured flight in every direction stayed at 0.00 degrees. Body-frame rays
therefore always pointed along fixed world axes, so "front" meant east whatever
direction the drone was flying. Here "front" is the HORIZONTAL direction from
the drone to its goal, which is defined even when the drone is hovering and is
exactly the direction in which the curriculum places its blocker.

Layout (azimuth relative to front, elevation), in observation order:
    0 front       (  0 deg,   0 deg)
    1 left        ( 90 deg,   0 deg)
    2 back        (180 deg,   0 deg)
    3 right       (270 deg,   0 deg)
    4 front_up    (  0 deg, +15 deg)
    5 front_down  (  0 deg, -15 deg)
"""
from __future__ import annotations

import numpy as np

from drone_env.defaults import LIDAR_RAY_LAYOUT

LIDAR_DIM = len(LIDAR_RAY_LAYOUT)
RAY_INDEX = {name: idx for idx, (name, _, _) in enumerate(LIDAR_RAY_LAYOUT)}

#REWARD SECTORS: FRONT = ALL THREE FORWARD RAYS; SIDES = ONE RAY EACH
FRONT_RAYS = np.array([RAY_INDEX["front"], RAY_INDEX["front_up"], RAY_INDEX["front_down"]], dtype=np.int32)
LEFT_RAYS = np.array([RAY_INDEX["left"]], dtype=np.int32)
RIGHT_RAYS = np.array([RAY_INDEX["right"]], dtype=np.int32)
BACK_RAYS = np.array([RAY_INDEX["back"]], dtype=np.int32)


#UNIT HORIZONTAL DIRECTION TO THE GOAL; +X WHEN THE GOAL IS DIRECTLY ABOVE/BELOW
def goal_heading(to_goal: np.ndarray) -> np.ndarray:
    flat = np.array([float(to_goal[0]), float(to_goal[1]), 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(flat))

    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return (flat / norm).astype(np.float32)


#WORLD-FRAME UNIT DIRECTION OF EVERY RAY, IN OBSERVATION ORDER
def ray_directions(to_goal: np.ndarray) -> np.ndarray:
    front = goal_heading(to_goal)
    left = np.array([-front[1], front[0], 0.0], dtype=np.float32)
    dirs = np.zeros((LIDAR_DIM, 3), dtype=np.float32)

    for idx, (_, azimuth_deg, elevation_deg) in enumerate(LIDAR_RAY_LAYOUT):
        az = np.deg2rad(float(azimuth_deg))
        el = np.deg2rad(float(elevation_deg))
        horizontal = np.cos(az) * front + np.sin(az) * left
        dirs[idx] = np.cos(el) * horizontal + np.array([0.0, 0.0, np.sin(el)], dtype=np.float32)

    return dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
