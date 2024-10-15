"""Stream images

"""

import airsim
import json
import numpy as np
import cv2 as cv
import os
import time
import shutil

from util.coordinates import quat_to_R, NED_2_ENU, airsim_to_nerfstudio

FOLDER_NAME = "unreal_moon_record"
DT = 0.1  # seconds
CV_MODE = False
STEREO = True

def get_cam_transform(cam_info):
    cam_pose = cam_info.pose
    position = cam_pose.position
    x, y, z = position.x_val, position.y_val, position.z_val
    q = cam_pose.orientation
    qx, qy, qz, qw = q.x_val, q.y_val, q.z_val, q.w_val

    transform = np.eye(4)
    airsim_R = quat_to_R([qx, qy, qz, qw])
    airsim_t = np.array([x, y, z])
    return airsim_to_nerfstudio((airsim_R, airsim_t))



if __name__ == "__main__":

    # Create output folder
    output_folder = f"output/{FOLDER_NAME}"
    if os.path.exists(output_folder):
        print("Data folder exists, deleting...")
        shutil.rmtree(output_folder)
    os.makedirs(f'{output_folder}/images')

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()

    frame_count = 0
    frames = []

    with open(f"{output_folder}/airsim_rec.txt", "a") as f:
        f.write("VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile\n")

    while True:
        try: 
            # Capture image
            img_path = f"images/img_{frame_count}.png"
            if CV_MODE:
                image = client.simGetImage(0, airsim.ImageType.Scene)
                image = cv.imdecode(np.frombuffer(image, np.uint8), -1)
                cv.imwrite(f'{output_folder}/{img_path}', image)
            elif STEREO:
                responses = client.simGetImages([airsim.ImageRequest("LeftCamera", airsim.ImageType.Scene),
                                                 airsim.ImageRequest("RightCamera", airsim.ImageType.Scene)])
                left_img_path = f"images/frame_{frame_count}_left.png"
                right_img_path = f"images/frame_{frame_count}_right.png"
                airsim.write_file(os.path.join(output_folder, left_img_path), responses[0].image_data_uint8)
                airsim.write_file(os.path.join(output_folder, right_img_path), responses[1].image_data_uint8)

                left_transform = get_cam_transform(client.simGetCameraInfo("LeftCamera"))
                right_transform = get_cam_transform(client.simGetCameraInfo("RightCamera"))
                left_frame = {
                    "file_path": left_img_path,
                    "transform_matrix": left_transform.tolist(),
                }
                frames.append(left_frame)
                right_frame = {
                    "file_path": right_img_path,
                    "transform_matrix": right_transform.tolist(),
                }
                frames.append(right_frame)

                print(f"Saved left and right images for frame {frame_count}")

            else:
                responses = client.simGetImages([airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene)])
                airsim.write_file(os.path.join(output_folder, img_path), responses[0].image_data_uint8)

                # Get camera pose
                cam_info = client.simGetCameraInfo("FrontCamera")
                cam_pose = cam_info.pose
                position = cam_pose.position
                x, y, z = position.x_val, position.y_val, position.z_val
                q = cam_pose.orientation
                qx, qy, qz, qw = q.x_val, q.y_val, q.z_val, q.w_val

                # Write to airsim_rec.txt
                with open(f"{output_folder}/airsim_rec.txt", "a") as f:
                    f.write(f"Car {frame_count} {x} {y} {z} {qw} {qx} {qy} {qz} img_{frame_count}\n")

                # TODO: Convert to nerfstudio format
                transform = np.eye(4)
                R = quat_to_R([qx, qy, qz, qw])
                R = NED_2_ENU @ R
                vx, vy, vz = R[:, 0], R[:, 1], R[:, 2]
                R = np.array([vy, -vz, -vx]).T

                t = np.array([x, y, z])
                t = NED_2_ENU @ t
                transform[:3, :3] = R
                transform[:3, 3] = t

                # Save camera pose
                frame = {
                    "file_path": img_path,
                    "transform_matrix": transform.tolist(),
                }
                frames.append(frame)
                
                print(f"Saved image {frame_count}")
            
            frame_count += 1
            time.sleep(DT)
        
        except KeyboardInterrupt:
            break
        

    # Get intrinsics from settings.json
    home_dir = os.path.expanduser("~")
    settings_path = os.path.join(home_dir, "Documents", "AirSim", "settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    # TODO: handle CVMODE and STEREO cases
    capture_settings = settings["Vehicles"]["Rover"]["Cameras"]["FrontCamera"]["CaptureSettings"][0]
    W = capture_settings["Width"]
    H = capture_settings["Height"]
    FOV = capture_settings["FOV_Degrees"]

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