import numpy as np
import xml.etree.ElementTree as ET
import pkg_resources


# ---- Parser for CF2X.URDF file ---- #
def parseURDFParameters(urdf_path):
    """Loads parameters from a URDF file."""

    URDF_TREE = ET.parse(
        pkg_resources.resource_filename("deepRL_for_autonomous_drones", urdf_path)
    ).getroot()

    M = float(URDF_TREE[1][0][1].attrib["value"])
    L = float(URDF_TREE[0].attrib["arm"])
    THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])
    IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
    IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
    IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
    J = np.diag([IXX, IYY, IZZ])
    J_INV = np.linalg.inv(J)
    KF = float(URDF_TREE[0].attrib["kf"])
    KM = float(URDF_TREE[0].attrib["km"])
    COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
    COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
    COLLISION_SHAPE_OFFSETS = [
        float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")
    ]
    COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
    MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
    GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
    PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
    DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
    DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
    DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
    DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
    DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
    DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])

    # print(
    #     "[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] mass {:f}, arm {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
    #         M,
    #         L,
    #         J[0, 0],
    #         J[1, 1],
    #         J[2, 2],
    #         KF,
    #         KM,
    #         THRUST2WEIGHT_RATIO,
    #         MAX_SPEED_KMH,
    #         GND_EFF_COEFF,
    #         PROP_RADIUS,
    #         DRAG_COEFF[0],
    #         DRAG_COEFF[2],
    #         DW_COEFF_1,
    #         DW_COEFF_2,
    #         DW_COEFF_3,
    #     )
    # )
    return (
        M,
        L,
        THRUST2WEIGHT_RATIO,
        J,
        J_INV,
        KF,
        KM,
        COLLISION_H,
        COLLISION_R,
        COLLISION_Z_OFFSET,
        MAX_SPEED_KMH,
        GND_EFF_COEFF,
        PROP_RADIUS,
        DRAG_COEFF,
        DW_COEFF_1,
        DW_COEFF_2,
        DW_COEFF_3,
    )
