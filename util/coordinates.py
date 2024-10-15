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


def euler_to_R(euler, seq='XYZ'):
    r = Rotation.from_euler(seq, euler, degrees=True)
    return r.as_matrix()


def airsim_to_nerfstudio(airsim_pose):
    """Convert rotation matrix from AirSim to NerfStudio coordinates

    Parameters
    ----------
    airsim_pose : tuple (R, t)

    Returns
    -------
    np.array (4 x 4)
        Nerfstudio transformation matrix

    """
    airsim_R, airsim_t = airsim_pose
    ns_R = NED_2_ENU @ airsim_R
    vx, vy, vz = ns_R[:, 0], ns_R[:, 1], ns_R[:, 2]
    ns_R = np.array([vy, -vz, -vx]).T

    ns_t = NED_2_ENU @ airsim_t
    T = np.eye(4)
    T[:3, :3] = ns_R
    T[:3, 3] = ns_t
    return T