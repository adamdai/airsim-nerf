import os
import numpy as np
import airsim
from airsim import Vector3r, Quaternionr, Pose
import plotly.graph_objects as go
import cv2 as cv
import json
import shutil

from util.coordinates import R_to_quat, airsim_to_nerfstudio, euler_to_R
from util.plotting import pose_traces


if __name__ == '__main__':

    PLOT_POSES = False
    FOLDER_NAME = 'landscape_mtns_path'

    #%% Spiral parameters

    # For LandscapeMountains
    start = np.array([177., -247., -33.])
    
    N = 20
    poses = []
    transforms = []
    airsim_t = start

    for i in range(N):
        airsim_t += np.array([0, -1, 0])
        airsim_R = euler_to_R([0, 0, -90])
        q = R_to_quat(airsim_R)

        poses.append((q, airsim_t.copy()))
        ns_pose = airsim_to_nerfstudio((airsim_R, airsim_t))
        T = np.eye(4)
        T[:3, :3] = ns_pose[0]
        T[:3, 3] = ns_pose[1]
        transforms.append(T)


    #%% Automated from here

    # Plot the spiral
    if PLOT_POSES:
        nerfstudio_poses = [(T[:3,:3], T[:3,3]) for T in transforms]
        poses_plot = pose_traces(nerfstudio_poses)
        fig = go.Figure(data=poses_plot)
        fig.update_layout(height=900, width=1600, scene=dict(aspectmode='data'))
        fig.show()

    # Create output folder
    output_folder = f"output/{FOLDER_NAME}"
    if os.path.exists(output_folder):
        print("Data folder exists, deleting...")
        shutil.rmtree(output_folder)
    os.makedirs(f'{output_folder}/images')

    frames = []

    client = airsim.VehicleClient()
    client.confirmConnection()

    for i, (q, position) in enumerate(poses):
        print(f"Capturing image {i}...")
        vehicle_pose = Pose(Vector3r(position[0], position[1], position[2]), q)
        client.simSetVehiclePose(vehicle_pose, True)
        
        # Capture image
        image = client.simGetImage(0, airsim.ImageType.Scene)
        image = cv.imdecode(np.frombuffer(image, np.uint8), -1)
        img_path = f'images/{i}.png'
        cv.imwrite(f'{output_folder}/{img_path}', image)

        # Save camera pose
        frame = {
            "file_path": img_path,
            "transform_matrix": transforms[i].tolist(),
            "colmap_im_id": i,
        }
        frames.append(frame)
    
    # Generate transforms.json
    # TODO: pull camera params from settings.json
    W, H = 720, 360
    FOV = 90
    fl = W / (2 * np.tan(np.radians(FOV) / 2))
    out = {
        "w": W,
        "h": H,
    }
    
    out["fl_x"] = fl
    out["fl_y"] = fl
    out["cx"] = W/2
    out["cy"] = H/2
    out["k1"] = 0.0
    out["k2"] = 0.0
    out["p1"] = 0.0
    out["p2"] = 0.0
        
    out["frames"] = frames

    print("Saving...", os.path.join(output_folder, 'transforms.json'))
    with open(os.path.join(output_folder, 'transforms.json'), 'w', encoding="utf-8") as f: 
        json.dump(out, f, indent=4)




