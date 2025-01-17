"""
Track a path of waypoints

"""

import airsim
import numpy as np
import time

from scipy.spatial.transform import Rotation


## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # start_unreal = np.array([39835.236, 62827.409, 15630.430])
    # goal_unreal = np.array([44570.0, -17780.0, 30930.0])
    # goal_airsim = (goal_unreal - start_unreal)[:2] / 100.0

    # Path
    # path_3d = np.load("data/moon_manual_1.npz")['states']
    path_3d = np.load("data/UnrealMoon_path_astar_1e2.npy")
    n_waypoints = len(path_3d)
    path = path_3d[1:,:2]  # (x,y) waypoints
    waypoint_index = 0
    waypoint = path[0]

    goal = path[-1]

    waypoint_thresh = 5.0  # meters
    goal_thresh = 10.0
    THROTTLE = 0.7

    # get vehicle position
    car_state = client.getCarState()
    #print("car state: %s" % car_state)


    try:
        print("driving routes")
        while(waypoint_index < n_waypoints):

            print(f"waypoint index: {waypoint_index} / {n_waypoints}")
            print(f"waypoint: {waypoint}")
            
            # Get car state
            car_state = client.getCarState()
            car_pose = client.simGetVehiclePose()
            # print("car state: %s" % car_state)
            x, y, z = car_pose.position.x_val, car_pose.position.y_val, car_pose.position.z_val
            qx, qy, qz, qw = car_pose.orientation.x_val, car_pose.orientation.y_val, car_pose.orientation.z_val, car_pose.orientation.w_val
            q = np.array([qx, qy, qz, qw])
            r = Rotation.from_quat(q)
            euler = r.as_euler('xyz')
            heading = euler[2]
            position_2d = np.array([x, y])

            # Compute relative angle to waypoint
            angle = np.arctan2(waypoint[1] - y, waypoint[0] - x)
            angle_diff = angle - heading
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi

            # Proportional steering control
            KP_STEER = 0.2
            steering = np.clip(KP_STEER * angle_diff, -1.0, 1.0)

            car_controls.steering = steering
            car_controls.throttle = THROTTLE
            client.setCarControls(car_controls)
            #time.sleep(1)

            if np.linalg.norm(position_2d - goal) < goal_thresh:
                car_controls.throttle = 0.0
                car_controls.brake = 1
                client.setCarControls(car_controls)
                break

            # If car is within 10 units of goal, break
            if np.linalg.norm(position_2d - waypoint) < waypoint_thresh:
                waypoint_index += 1
                waypoint = path[waypoint_index]


    except KeyboardInterrupt:
        # Restore to original state
        client.reset()
        client.enableApiControl(False)


    # # Restore to original state
    # client.reset()
    # client.enableApiControl(False)

    


