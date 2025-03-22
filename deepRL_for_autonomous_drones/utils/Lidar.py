import math
import pybullet as p
import numpy as np


class Lidar:
    def __init__(self):
        self.replace_item_uniqueIds = None
        p.removeAllUserDebugItems()

    # ---- Rotation matrix representing orientation ----#
    def rotation_matrix(self, roll, pitch, yaw):
        rot_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)],
            ]
        )

        rot_y = np.array(
            [
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)],
            ]
        )

        rot_z = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        return rot_z @ rot_y @ rot_x

    def CheckHits(
        self,
        ray_from_position,
        ray_orientation,
        ray_length,
        offset,
        launch_pad_id,
        draw_debug_line=False,
    ):
        """
        Casts rays in a full 3D spherical pattern and returns the ray test results

        Parameters
        ---
        ray_from_position : list
            List storing the starting position of the ray [x, y, z]
        ray_orientation : list
            Orientation of the ray source [roll, pitch, yaw]
        ray_length : float
            Length of the ray
        offset : float
            Offset from rayFromPosition
        draw_debug_line : bool
            Whether to draw the ray

        Returns
        ---
        hit_results: tuple
            Results of the ray casting
        """

        # ---- Compute rotation matrix to transform from sensor local coordinates to world coordinates ----#
        R = self.rotation_matrix(
            ray_orientation[0], ray_orientation[1], ray_orientation[2]
        )

        # ---- Vertical range for the scan (-90 degress to 90 degrees elevation)
        elev_min = -math.pi / 2
        elev_max = math.pi / 2

        ray_from = []
        ray_to = []

        # ---- Number of rays. Becomes 36 rays in 3D space ----#
        num_horizontal = 6
        num_vertical = 6

        # ---- Generate 3D ray directions ----#
        for i in range(num_vertical):
            if num_vertical > 1:
                elev = elev_min + i * (elev_max - elev_min) / (num_vertical - 1)
            else:
                elev = 0

            for j in range(num_horizontal):
                azimuth = (2 * math.pi * j) / num_horizontal

                # ---- Spherical to Cartesian conversion in the sensor's local frame ----#
                local_direction = [
                    math.cos(elev) * math.cos(azimuth),
                    math.cos(elev) * math.sin(azimuth),
                    math.sin(elev),
                ]

                # ---- Transform the direction vector into world coordinates ----#
                world_dir = R @ local_direction

                # ---- Compute start and end points for the ray ----#
                start_pos = np.array(ray_from_position) + offset
                end_pos = start_pos + ray_length * world_dir

                ray_from.append(start_pos)
                ray_to.append(end_pos)

        # ---- Perform ray casting ----#
        raw_hit_results = p.rayTestBatch(ray_from, ray_to, numThreads=0)

        hit_results = []
        for res in raw_hit_results:
            if res[0] != launch_pad_id:
                hit_results.append(res)
            else:
                hit_results.append(
                    (-1, 0, 1.0, ray_to[raw_hit_results.index(res)])
                )  # No hit, return max distance

        # ---- Optionally draw the rays ----#
        if draw_debug_line:
            if self.replace_item_uniqueIds is None:
                self.replace_item_uniqueIds = []
                for idx, res in enumerate(hit_results):
                    if res[0] != -1:
                        # ---- Hit detected: draw ray from sensor to hit point (red) ----#
                        debug_id = p.addUserDebugLine(
                            ray_from[idx], res[3], lineColorRGB=[1, 0, 0]
                        )
                    else:
                        # ---- No hit: draw ray from sensor to end point (green) ----#
                        debug_id = p.addUserDebugLine(
                            ray_from[idx], ray_to[idx], lineColorRGB=[0, 1, 0]
                        )
                    self.replace_item_uniqueIds.append(debug_id)
            else:
                # ---- Update existing debug lines ----#
                for idx, res in enumerate(hit_results):
                    if res[0] != -1:
                        p.addUserDebugLine(
                            ray_from[idx],
                            res[3],
                            lineColorRGB=[1, 0, 0],
                            replaceItemUniqueId=self.replace_item_uniqueIds[idx],
                        )
                    else:
                        p.addUserDebugLine(
                            ray_from[idx],
                            ray_to[idx],
                            lineColorRGB=[0, 1, 0],
                            replaceItemUniqueId=self.replace_item_uniqueIds[idx],
                        )
        return hit_results
