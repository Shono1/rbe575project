from __future__ import annotations
import matplotlib.pyplot as plt
import pickle as pkl
from spatialmath import SE3
import numpy as np
import roboticstoolbox as rtb

with open('ts_record.pkl', 'rb') as f:
    ts_points: list[SE3] = pkl.load(f)

with open('js_traj.pkl', 'rb') as f:
    js_traj = pkl.load(f)

dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

desired_data = [robot.fkine(qs) for qs in js_traj]
desired_data = np.array([pt.t for pt in desired_data])

cbf_data = np.array([pt.t for pt in ts_points])
plt.scatter(cbf_data[:, 0], cbf_data[:, 1])
plt.scatter(desired_data[:, 0], desired_data[:, 1])
plt.show()
