import random
from importlib.resources import files
import pybullet as p

import numpy as np


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
