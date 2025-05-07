# from __future__ import annotations
# import matplotlib.pyplot as plt
# import pickle as pkl
# from spatialmath import SE3
# import numpy as np
# import roboticstoolbox as rtb

# with open('rbe575project/lib/projectcode/trajectories/ts_traj_nocbf.pkl', 'rb') as f:
#     ts_points: list[SE3] = pkl.load(f)

# dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
#               rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
#               rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
#               rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
# robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

# cbf_data = np.array(ts_points)
# plt.scatter(cbf_data[:, 0], cbf_data[:, 1])
# plt.xlim([-200, 200])
# plt.ylim([-200, 200])

# with open('rbe575project/lib/projectcode/trajectories/js_traj_nocbf.pkl', 'rb') as f:
#     js_record = np.array(pkl.load(f))
# plt.subplots(4, 1)
# for i in range(1, 5):
#     plt.subplot(4, 1, i)
#     plt.plot(np.linspace(0, 10, len(js_record)), js_record[:, i-1])
# plt.show()

import os
import matplotlib.pyplot as plt
import pickle as pkl
from spatialmath import SE3
import numpy as np
import roboticstoolbox as rtb

# --- Setup robot ---
dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
          rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
          rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
          rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

# --- Task Space Trajectories ---
ts_dir = 'rbe575project/lib/projectcode/ts_traj'
ts_files = [f for f in os.listdir(ts_dir) if f.endswith('.pkl')]

plt.figure()
for f in ts_files:
    with open(os.path.join(ts_dir, f), 'rb') as file:
        ts_points: list[SE3] = pkl.load(file)
        cbf_data = np.array(ts_points)
        plt.scatter(cbf_data[:, 0], cbf_data[:, 1], label=f)

plt.xlim([-200, 200])
plt.ylim([-200, 200])
plt.legend()
plt.title("Task Space Trajectories")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")

# --- Joint Space Trajectories ---
js_dir = 'rbe575project/lib/projectcode/js_traj'
js_files = [f for f in os.listdir(js_dir) if f.endswith('.pkl')]

fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
for f in js_files:
    with open(os.path.join(js_dir, f), 'rb') as file:
        js_record = np.array(pkl.load(file))
        time = np.linspace(0, 10, len(js_record))
        for i in range(4):
            axs[i].plot(time, js_record[:, i], label=f)

for i, ax in enumerate(axs):
    ax.set_ylabel(f'Joint {i+1} (rad)')
    ax.legend()
axs[-1].set_xlabel('Time (s)')
fig.suptitle("Joint Space Trajectories")

plt.tight_layout()
plt.show()
