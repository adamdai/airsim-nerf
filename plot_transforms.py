import json
import numpy as np
import plotly.graph_objects as go

from util.plotting import pose_traces


transforms_path = 'output/unreal_moon/transforms.json'
# transforms_path = '../../NeRF/nerfstudio/data/AirSim/unreal_moon_cv_fly_1/transforms.json'
with open(transforms_path, 'r') as f:
    transforms = json.load(f)

poses = []
for frame in transforms['frames']:
    T = np.array(frame['transform_matrix'])
    R, t = T[:3, :3], 5.0 * T[:3, 3]
    poses.append((R, t))

poses = pose_traces(poses)


# Create and show the plot with all the traces
fig = go.Figure(data=poses)
fig.update_layout(height=900, width=1600, scene=dict(aspectmode='data'))
fig.show()