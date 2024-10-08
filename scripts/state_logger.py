"""Log vehicle pose

"""

import numpy as np
import airsim
import time

from scipy.spatial.transform import Rotation as R

def quat_to_R(quat):
    """Quaternion in scalar-last (x, y, z, w) format"""
    r = R.from_quat(quat)
    return r.as_matrix()

client = airsim.MultirotorClient()
client.confirmConnection()

dt = 0.1
states = []
collision_count = 0

try:
    while True:
        pose = client.simGetVehiclePose()
        x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val
        
        pose = client.simGetCameraInfo(0).pose
        qx, qy, qz, qw = pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val

        rot = quat_to_R([qx, qy, qz, qw])
        print(f"{np.round(rot, 2)}")
        
        position = np.array([x, y, z])
        states.append(position)
        # print(f"x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")
        time.sleep(dt)

except KeyboardInterrupt:
    print("Saving data")
    timestamp = time.strftime("%Y%m%d-%H%M")
    np.savez(f'../data/states_{timestamp}.npz', states=states)
    print("Interrupted by user, shutting down")
    client.enableApiControl(False)
    exit()