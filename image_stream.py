"""Stream images

"""

import airsim
import cv2
import numpy as np
import os
import time

if __name__ == "__main__":
    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    # brake the car
    car_controls.brake = 0
    car_controls.throttle = 0
    client.setCarControls(car_controls)

    # Get an image
    start_time = time.perf_counter()
    
    # responses = client.simGetImages([airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene)])
    # airsim.write_file("/tmp/airsim_images/img.png", responses[0].image_data_uint8)

    responses = client.simGetImages([airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene), 
                                     airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
    airsim.write_file("/tmp/airsim_images/img.png", responses[0].image_data_uint8)
    depth_float = np.array(responses[1].image_data_float)
    depth_float = depth_float.reshape(responses[1].height, responses[1].width)
    data_normalized = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX)
    data_uint8 = data_normalized.astype(np.uint8)
    cv2.imwrite("/tmp/airsim_images/depth.png", data_uint8)

    # responses = client.simGetImages([airsim.ImageRequest("Depth", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
    # image = client.simGetImage("FrontCamera", airsim.ImageType.Scene)
    # image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

    # responses = client.simGetImages([
    #     airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
    #     airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
    #     airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
    #     airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
    
    print(f"Getting images took: {time.perf_counter() - start_time} seconds")

    start_time = time.perf_counter()

    # tmp_dir = "/tmp/airsim_images"
    # for response_idx, response in enumerate(responses):
    #     filename = os.path.join(tmp_dir, f"{0}_{response.image_type}_{response_idx}")

    #     if response.pixels_as_float:
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
    #         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    #     elif response.compress: #png format
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    #         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    #     else: #uncompressed array
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    #         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
    #         img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
    #         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

    print(f"Saving images took: {time.perf_counter() - start_time} seconds")