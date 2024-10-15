import os
import numpy as np
import airsim
from airsim import Vector3r, Quaternionr, Pose
import plotly.graph_objects as go
import cv2 as cv
import json
import shutil

from util.coordinates import R_to_quat, NED_2_ENU, XY_ROT_180
from util.plotting import pose_traces


def generate_spiral(center, num_rings, radii, heights, num_points):
    """General spiral of camera poses in nerfstudio coordinates"""
    airsim_poses = []
    nerfstudio_transforms = []  # camera transform matrices for nerfstudio

    for i in range(num_rings):
        r = radii[i]
        z = heights[i]
        n = num_points[i]

        for j in range(n):
            # AirSim coordinates
            angle = 2*np.pi*j/n
            x = r*np.cos(angle)
            y = r*np.sin(angle)
            offset = np.array([x, y, -z])
            position = center + offset

            vx = -offset  # ray pointing from camera to center ("Forward" vector)
            vx = vx / np.linalg.norm(vx)  # normalize
            d = np.linalg.norm(vx[:2])    # distance in x-y plane
            unit_z = vx[2]                # z component of unit vector                    
            vz = np.array([-(unit_z/d)*vx[0], -(unit_z/d)*vx[1], d])  # "Down" vector
            vy = np.cross(vz, vx)                                     # "Right" vector
            airsim_R = np.vstack((vx, vy, vz)).T
            q = R_to_quat(airsim_R)
            airsim_poses.append((position, q))

            # Nerfstudio coordinates
            R = NED_2_ENU @ airsim_R
            vx, vy, vz = R[:, 0], R[:, 1], R[:, 2]
            R = np.array([vy, -vz, -vx]).T
            t = NED_2_ENU @ offset
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            nerfstudio_transforms.append(T)

    return airsim_poses, nerfstudio_transforms


if __name__ == '__main__':

    PLOT_POSES = False
    FOLDER_NAME = 'landscape_mtns_spiral'

    #%% Spiral parameters

    # For LandscapeMountains
    center = np.array([99., -449., -57.])
    num_rings = 3
    radii = [250, 300, 400]
    heights = [150, 180, 200]
    num_points = [20, 20, 20]

    # For Moon
    # unreal_start = np.array([5184.151855, -182487.484375, 1295.814209])
    # unreal_goal = np.array([110060.0, -148820.0, 25740.0])
    # x = (unreal_goal - unreal_start) / (2 * 100)

    # center = np.array([x[0], x[1], 0.])
    # num_rings = 5
    # radii = [700, 600, 500, 400, 500]
    # heights = [150, 250, 350, 450, 600]
    # num_points = [20, 20, 20, 20, 20]

    #%% Automated from here

    poses, transforms = generate_spiral(center, num_rings, radii, heights, num_points)

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

    for i, (position, q) in enumerate(poses):
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
        
        #airsim.time.sleep(0.01)
    
    # Generate transforms.json
    # TODO: pull camera params from settings.json
    W, H = 1920, 1080
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




