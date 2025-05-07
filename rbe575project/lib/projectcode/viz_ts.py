import os
import matplotlib.pyplot as plt
import pickle as pkl
from spatialmath import SE3
import numpy as np
import roboticstoolbox as rtb

dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
          rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
          rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
          rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

ts_dir = 'rbe575project/lib/projectcode/ts_traj'
ts_files = [f for f in os.listdir(ts_dir) if f.endswith('.pkl')]

plt.figure()
for f in ts_files:
    with open(os.path.join(ts_dir, f), 'rb') as file:
        ts_points: list[SE3] = pkl.load(file)
        cbf_data = np.array(ts_points)
        plt.scatter(cbf_data[:, 0], cbf_data[:, 1], label=f)

plt.xlim([-50, 200])
plt.ylim([-300, 300])
plt.legend(prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Task Space Trajectories", fontsize=20)
plt.xlabel("X (mm)", fontsize=20)
plt.ylabel("Y (mm)", fontsize=20)

js_dir = 'rbe575project/lib/projectcode/js_traj'
js_files = [f for f in os.listdir(js_dir) if f.endswith('.pkl')]

joint_count = 4

# Loop through each joint
for joint_idx in range(joint_count):
    plt.figure(figsize=(8, 4))
    for f in js_files:
        with open(os.path.join(js_dir, f), 'rb') as file:
            js_record = np.array(pkl.load(file))
            time = np.linspace(0, 10, len(js_record))
            plt.plot(time, js_record[:, joint_idx], label=f, linewidth=3.5)

    plt.title(f"Joint {joint_idx + 1} Trajectory", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel(f"Joint {joint_idx + 1} (rad)", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()

cbf_dir = 'rbe575project/lib/projectcode/cbf_functions'
cbf_files = [f for f in os.listdir(cbf_dir) if f.endswith('.pkl')]

plt.figure(figsize=(8, 4))
for f in cbf_files:
    with open(os.path.join(cbf_dir, f), 'rb') as file:
        cbf_record = np.array(pkl.load(file))
        time = np.linspace(0, 10, len(cbf_record))
        plt.plot(time, cbf_record[:], label=f, linewidth=3.5)

    plt.title(f"Determinant Threshold", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel(f"Determinant", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()

plt.show()
