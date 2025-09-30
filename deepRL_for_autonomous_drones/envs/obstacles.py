import random
from importlib.resources import files
import pybullet as p

# import pkg_resources
import numpy as np


# def generateStaticTrees(fixed_tree_positions, fixed_tree_types, pyb_client):
#     trees = []
#     for pos, tree_type in zip(fixed_tree_positions, fixed_tree_types):
#         trees.append(
#             pyb_client.loadURDF(
#                 # pkg_resources.resource_filename("deepRL_for_autonomous_drones", tree_type),
#                 str(files("deepRL_for_autonomous_drones") / tree_type),
#                 basePosition=pos,
#                 useFixedBase=True,
#             )
#         )

#     return trees

#------- Seed / layout_pool -------#
def generateStaticTrees(fixed_tree_positions, fixed_tree_types, pyb_client):
    trees = []
    for pos, tree_type in zip(fixed_tree_positions, fixed_tree_types):
        random_uniform = random.uniform(0.5, 2.0)
        trees.append(
            pyb_client.loadURDF(
                str(files("deepRL_for_autonomous_drones") / tree_type),
                basePosition=pos,
                useFixedBase=True,
                globalScaling=random_uniform,
            )
        )

    return trees


#------- PARAMETRIC TREES ---------#
# def generateParametricTrees(positions, specs, bc):
#     """
#     canopy_r <= 0 disables canopy; trunks only unless canopy_r > 0.
#     Trunks are BOXES with square cross-section: width=depth=2*trunk_r, height=trunk_h.
#     """
#     uids = []
#     for (x, y, z), s in zip(positions, specs):
#         tr = float(s.get("trunk_r", 0.20))
#         th = float(s.get("trunk_h", 5.0))

#         trunk_rgba = s.get("trunk_rgba", [0.35, 0.20, 0.10, 1.0])

#         # BOX trunk: half extents = [w/2, d/2, h/2] = [tr, tr, th/2]
#         trunk_col = bc.createCollisionShape(p.GEOM_BOX, halfExtents=[tr, tr, th / 2.0])
#         trunk_vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=[tr, tr, th / 2.0], rgbaColor=trunk_rgba)

#         uid = bc.createMultiBody(
#             baseMass=0.0,
#             baseCollisionShapeIndex=trunk_col,
#             baseVisualShapeIndex=trunk_vis,
#             basePosition=[x, y, z + th / 2.0],
#             baseOrientation=[0, 0, 0, 1],
#         )
#         bc.changeDynamics(uid, -1, lateralFriction=0.9, restitution=0.0)

#         # trees group=2, mask=4 (drone only)
#         # bc.setCollisionFilterGroupMask(uid, -1, 2, 4)
#         uids.append(uid)
#     return uids


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
