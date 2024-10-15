import numpy as np 
from scipy.spatial.transform import Rotation

from airsim import Vector3r, Quaternionr, Pose

NED_2_ENU = np.array([[0, 1, 0], 
                      [1, 0, 0], 
                      [0, 0, -1]])

XY_ROT_180 = np.array([[-1, 0, 0], 
                       [0, -1, 0], 
                       [0, 0, 1]])

def quat_to_R(q):
    """Convert quaternion to 3D rotation matrix 

    Parameters
    ----------
    q : np.array (1 x 4)
        Quaternion in scalar-last (x, y, z, w) format

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix

    """
    if type(q) is Quaternionr:
        q = np.array([q.x_val, q.y_val, q.z_val, q.w_val])
    r = Rotation.from_quat(q)
    return r.as_matrix()


def R_to_quat(R):
    """Convert 3D rotation matrix to quaternion

    Parameters
    ----------
    R : np.array (3 x 3)
        Rotation matrix

    Returns
    -------
    np.array (1 x 4)
        Quaternion in scalar-last (x, y, z, w) format

    """
    r = Rotation.from_matrix(R)
    q = r.as_quat()
    return Quaternionr(q[0], q[1], q[2], q[3])


def airsim_to_nerfstudio(airsim_R):
    """Convert rotation matrix from AirSim to NerfStudio coordinates

    Parameters
    ----------
    R : np.array (3 x 3)
        Rotation matrix in AirSim coordinates

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix in NerfStudio coordinates

    """
    NED_2_NERFSTUDIO = np.array([[0, 0, -1],
                                 [1, 0, 0],
                                 [0, -1, 0]])  # or inverse?
    nerfstudio_R = np.eye(3)  # TODO
    return nerfstudio_R