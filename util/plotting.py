import numpy as np
import plotly.graph_objects as go

def pose_trace(pose):
    """Create a trace for a pose (R, t)"""
    # Unpack pose into rotation matrix R and translation vector t
    R, t = pose
    t = np.array(t)  # Ensure t is a numpy array
    
    # Define arrow colors for each axis (RGB)
    colors = ['red', 'green', 'blue']
    
    # Define the unit vectors from the columns of R
    axis_vectors = [R[:, 0], R[:, 1], R[:, 2]]
    
    # Create traces for each axis (X, Y, Z)
    traces = []
    for i, vec in enumerate(axis_vectors):
        arrow_start = t
        arrow_end = t + vec  # Arrow points in the direction of the column of R
        
        # Create an arrow trace for the axis
        trace = go.Scatter3d(
            x=[arrow_start[0], arrow_end[0]],
            y=[arrow_start[1], arrow_end[1]],
            z=[arrow_start[2], arrow_end[2]],
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color=colors[i], width=5),
            showlegend=False
        )
        traces.append(trace)
    
    return traces

def pose_traces(pose_list):
    """Create traces for a list of poses
    
    pose_list : list of tuples
        List of poses, where each pose is a tuple (R, t)
    
    """
    all_traces = []

    for pose in pose_list:
        traces = pose_trace(pose)
        all_traces.extend(traces)  
    
    return all_traces