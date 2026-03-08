from dynamics.differential_drive import step
from controller.mpc_cbf_controller import controller

state = [0.0, 0.0, 0.0]
dt = 0.1

for k in range(20):
    ref = None
    d_min = 2.0

    u = controller(state, ref, d_min, mode="mpc")
    state = step(state, u, dt)

    print(f"t={k*dt:.1f}s | state={state}")
