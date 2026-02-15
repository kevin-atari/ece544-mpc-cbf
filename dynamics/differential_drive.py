import numpy as np

def step(state, control, dt):
    """
    Differential-drive (unicycle) model

    state: (x, y, theta)
    control: (v, omega)
    dt: timestep
    """
    x, y, theta = state
    v, omega = control

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    return np.array([x_next, y_next, theta_next])
