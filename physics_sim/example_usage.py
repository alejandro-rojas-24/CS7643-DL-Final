from projectiles import ProjectileSimulator
import numpy as np

def ground_function(x):
    return 100 * np.sin(0.01 * x)

# init simulator
sim = ProjectileSimulator()
sim.add_ground(-1000, 1000, 10, ground_function)

# add projectile
projectile_body = sim.add_projectile((0, 100), (500, 400))

# simulate
positions = sim.simulate(duration=5)

# plot trajectory
sim.plot_trajectory(positions, ground_function, -1000, 1000, 10)