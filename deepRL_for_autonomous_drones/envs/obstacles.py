import random
from importlib.resources import files
import pybullet as p

# import pkg_resources
import numpy as np


def getFixedTreePositions():
    fixed_tree_positions = [
        # ---- Top left (-X, Y) Quadrant ----#
        (-1, 1, 0),
        (-3, 1, 0),
        (-5, 1, 0),
        (-2, 2, 0),
        (-4, 2, 0),
        (-1, 3, 0),
        (-3, 3, 0),
        (-5, 3, 0),
        (-2, 4, 0),
        (-4, 4, 0),
        (-1, 5, 0),
        (-3, 5, 0),
        (-5, 5, 0),
        (-2, 6, 0),
        (-4, 6, 0),
        # ---- Top right (X, Y) Quadrant ----#
        (1, 1, 0),
        (3, 1, 0),
        (5, 1, 0),
        (2, 2, 0),
        (4, 2, 0),
        (1, 3, 0),
        (3, 3, 0),
        (5, 3, 0),
        (2, 4, 0),
        (4, 4, 0),
        (1, 5, 0),
        (3, 5, 0),
        (5, 5, 0),
        (2, 6, 0),
        (4, 6, 0),
        # ---- Bottom right (X, -Y) Quadrant ----#
        (1, -1, 0),
        (3, -1, 0),
        (5, -1, 0),
        (2, -2, 0),
        (4, -2, 0),
        (1, -3, 0),
        (3, -3, 0),
        (5, -3, 0),
        (2, -4, 0),
        (4, -4, 0),
        (1, -5, 0),
        (3, -5, 0),
        (5, -5, 0),
        (2, -6, 0),
        (4, -6, 0),
        # ---- Bottom left (-X, -Y) Quadrant ----#
        (-1, -1, 0),
        (-3, -1, 0),
        (-5, -1, 0),
        (-2, -2, 0),
        (-4, -2, 0),
        (-1, -3, 0),
        (-3, -3, 0),
        (-5, -3, 0),
        (-2, -4, 0),
        (-4, -4, 0),
        (-1, -5, 0),
        (-3, -5, 0),
        (-5, -5, 0),
        (-2, -6, 0),
        (-4, -6, 0),
        # ---- Along X axis ----#
        (-6, 0, 0),
        (-4, 0, 0),
        (-2, 0, 0),
        (2, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
        # ---- Along Y axis ----#
        (0, 6, 0),
        (0, 4, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, -4, 0),
        (0, -6, 0),
    ]

    return fixed_tree_positions


def generateStaticTrees(fixed_tree_positions, fixed_tree_types, pyb_client):
    trees = []
    for pos, tree_type in zip(fixed_tree_positions, fixed_tree_types):
        trees.append(
            pyb_client.loadURDF(
                # pkg_resources.resource_filename("deepRL_for_autonomous_drones", tree_type),
                str(files("deepRL_for_autonomous_drones") / tree_type),
                basePosition=pos,
                useFixedBase=True,
            )
        )

    return trees


def loadStaticBlocks():
    static_blocks = []
    static_blocks.append(
        p.loadURDF(
            # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/static_blocks.urdf"),
            str(files("deepRL_for_autonomous_drones") / "assets/static_blocks.urdf"),
            basePosition=[3, 3, 3],
            useFixedBase=True,
        )
    )
    static_blocks.append(
        p.loadURDF(
            # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/static_blocks.urdf"),
            str(files("deepRL_for_autonomous_drones") / "assets/static_blocks.urdf"),
            basePosition=[3, -3, 3],
            useFixedBase=True,
        )
    )
    static_blocks.append(
        p.loadURDF(
            # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/static_blocks.urdf"),
            str(files("deepRL_for_autonomous_drones") / "assets/static_blocks.urdf"),
            basePosition=[-3, 3, 3],
            useFixedBase=True,
        )
    )
    static_blocks.append(
        p.loadURDF(
            # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/static_blocks.urdf"),
            str(files("deepRL_for_autonomous_drones") / "assets/static_blocks.urdf"),
            basePosition=[-3, -3, 3],
            useFixedBase=True,
        )
    )

    return static_blocks


def loadMovingBlocks():
    first_moving_block = p.loadURDF(
        # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/moving_blocks.urdf"),
        str(files("deepRL_for_autonomous_drones") / "assets/moving_blocks.urdf"),
        basePosition=[0, 0, 1],
        useFixedBase=True,
    )
    second_moving_block = p.loadURDF(
        # pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/moving_blocks.urdf"),
        str(files("deepRL_for_autonomous_drones") / "assets/moving_blocks.urdf"),
        basePosition=[0, 0, 1],
        useFixedBase=True,
    )

    return first_moving_block, second_moving_block


def loadTorusObstacles():
    toruses = []
    torus_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        # fileName=pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/torus.obj"),
        fileName=str(files("deepRL_for_autonomous_drones") / "assets/torus.obj"),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    torus_visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        # fileName=pkg_resources.resource_filename("deepRL_for_autonomous_drones", "assets/torus.obj"),
        fileName=str(files("deepRL_for_autonomous_drones") / "assets/torus.obj"),
        rgbaColor=[1, 0, 0, 1],
    )
    torus_id_one = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=torus_collision,
        baseVisualShapeIndex=torus_visual,
        basePosition=[0, 0, 1],
        baseOrientation=[1, 1, 1, 1],
    )
    toruses.append(torus_id_one)

    torus_id_two = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=torus_collision,
        baseVisualShapeIndex=torus_visual,
        basePosition=[0, 0, 2],
        baseOrientation=[1, 1, 1, 1],
    )
    toruses.append(torus_id_two)

    return toruses
