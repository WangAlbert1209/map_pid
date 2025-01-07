
import numpy as np
from numpy import random

initial_angles = np.linspace(0, 2*np.pi, 5)
initial_angles=initial_angles[:4]
initial_angles=initial_angles%(2*np.pi)
for i in range(len(initial_angles)):
    if initial_angles[i]>np.pi:
        initial_angles[i]=initial_angles[i]-2*np.pi