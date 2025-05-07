from __future__ import annotations
import matplotlib.pyplot as plt
import pickle as pkl
from spatialmath import SE3
import numpy as np
import roboticstoolbox as rtb

with open('rbe575project/lib/projectcode/trajectories/ts_traj_nocbf.pkl', 'rb') as f:
    ts_points: list[SE3] = pkl.load(f)

dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

cbf_data = np.array(ts_points)
plt.scatter(cbf_data[:, 0], cbf_data[:, 1])
plt.xlim([-200, 200])
plt.ylim([-200, 200])

with open('rbe575project/lib/projectcode/trajectories/js_traj_nocbf.pkl', 'rb') as f:
    js_record = np.array(pkl.load(f))
plt.subplots(4, 1)
for i in range(1, 5):
    plt.subplot(4, 1, i)
    plt.plot(np.linspace(0, 10, len(js_record)), js_record[:, i-1])
plt.show()