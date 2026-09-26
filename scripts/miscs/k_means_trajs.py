from pathlib import Path as _Path
import os as _os
_PROJECT_ROOT = _Path(_os.environ.get("NAVSIM_DEVKIT_ROOT", str(_Path(__file__).resolve().parents[2])))

import os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from matplotlib.cm import get_cmap

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from tqdm import tqdm
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

SPLIT = "trainval"  # ["mini", "test", "trainval"]
FILTER = "navtrain"  # ["navtrain", "navtest", "all_scenes", ]
num_poses = 8 # 0.5s * 8 = 4s
K = 256

"""
save navtrain future trajectories as numpy array
"""
hydra.initialize(config_path="../../navsim/planning/script/config/common/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(f"{_PROJECT_ROOT}/dataset/")

scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
        # sensor_config=SensorConfig.build_all_sensors(),
)

future_trajectories_list = []

def process_token(token):
        scene = scene_loader.get_scene_from_token(token)
        future_trajectory = scene.get_future_trajectory(
        num_trajectory_frames=num_poses,
        ).poses
        return future_trajectory

print("Collecting future trajectories...")
for token in tqdm(scene_loader.tokens):
        scene = scene_loader.get_scene_from_token(token)
        future_trajectory = scene.get_future_trajectory(
                        num_trajectory_frames=num_poses, 
                ).poses
        future_trajectories_list.append(future_trajectory)

# save future_trajectories_list as numpy array
numpy_path = f"future_trajectories_list_{SPLIT}_{FILTER}.npy"
np.save(numpy_path, future_trajectories_list)

# load 
future_trajectories_list = np.load(f"{_PROJECT_ROOT}/extra_data/future_trajectories_list_trainval_navtrain.npy")
np.set_printoptions(suppress=True)
N = len(future_trajectories_list)
future_trajectories_array = np.array(future_trajectories_list)  # (N, 2), the last position
flattened_trajectories = future_trajectories_array.reshape(N, -1).astype(np.float32)  # (N, 24)

kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1000)
kmeans.fit(flattened_trajectories)

labels = kmeans.labels_
trajectory_anchors = kmeans.trajectory_anchors_


trajectory_anchors = trajectory_anchors.reshape(K, 8, 3)

# save trajectory_anchors as numpy array
numpy_path = f"{_PROJECT_ROOT}/extra_data/planning_vb/trajectory_anchors_{K}.npy"
np.save(numpy_path, trajectory_anchors)

""""
Visual code
"""
numpy_path = f"{_PROJECT_ROOT}/extra_data/planning_vb/trajectory_anchors_{K}.npy"
trajectory_anchors = np.load(numpy_path)

# Visualize all cluster centers on a single plot
fig, ax = plt.subplots(figsize=(15, 15))
cmap = get_cmap('hsv', K)  # Use colormap to distinguish between different trajectories

for i in range(K):
        trajectory = trajectory_anchors[i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color=cmap(i), label=f'Cluster {i}', alpha=0.6, linewidth=1.5)

ax.set_title('All Cluster Centers')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(False)
plt.tight_layout()
plt.savefig(f'{_PROJECT_ROOT}/extra_data/planning_vb/trajectory_anchors_{K}_no_grid.png')

# # save trajectory_anchors as numpy array
# Load cluster centers data
numpy_path = f"{_PROJECT_ROOT}/extra_data/planning_vb/trajectory_anchors_{K}.npy"
trajectory_anchors = np.load(numpy_path)

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(15, 15))

highlight_idx = 57  # Choose the trajectory to highlight
cmap = get_cmap('hsv', K)  # Use colormap for distinguishing if needed

# Convert RGB (115, 137, 177) to a normalized value in [0, 1]
background_color = (115/255, 137/255, 177/255)

# Plot each trajectory
for i in range(K):
    trajectory = trajectory_anchors[i]
    if i == highlight_idx:
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label=f'Highlighted Cluster {i}', alpha=0.9, linewidth=5)
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=background_color, alpha=0.9, linewidth=5)

# Set plot properties
ax.set_title('Highlighted Cluster with Background Clusters')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend(loc='upper right')
ax.grid(False)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'{_PROJECT_ROOT}/vis/trajectory_anchors_{K}_highlighted_{highlight_idx}.png')
print(f"Saved figure to {_PROJECT_ROOT}/vis/trajectory_anchors_{K}_highlighted_{highlight_idx}.png")

