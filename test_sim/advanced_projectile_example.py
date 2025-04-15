# from projectiles import AdvancedProjectileSimulator
# import numpy as np
# import matplotlib.pyplot as plt


# def orbital_ground(x):
#     return 0.5 * x * np.sin(x / 1000)


# # Complex terrain with multiple sinusoidal components
# def complex_terrain(x):
#     return 100 * (
#         0.6 * np.sin(0.01 * x) + 0.3 * np.sin(0.03 * x) + 0.1 * np.sin(0.06 * x)
#     )


# # Initialize simulator with realistic air density
# sim = AdvancedProjectileSimulator(
#     air_density=1.225,  # Sea level air density
#     drag_coeff=0.47,  # Sphere drag coefficient
# )

# # Create complex ground
# sim.add_ground(-1000, 1000, 10, complex_terrain, friction=0.7)

# # Add projectile with high initial velocity
# projectile = sim.add_projectile(
#     position=(0, 300),  # Initial position (cm)
#     velocity=(800, 400),  # Initial velocity (cm/s)
#     radius=5,  # 5cm radius
#     elasticity=0.8,
# )

# # Simulate for 5 seconds
# positions = sim.simulate(5)

# # Plot results

# sim.plot_trajectory(positions, complex_terrain, -1000, 1000, 10)
# plt.title("Projectile Trajectory with Aerodynamic Drag")
# plt.show()

from projectiles import ProjectileSimulator, AdvancedProjectileSimulator
import numpy as np


def ground_function(x):
    return 100 * np.sin(0.01 * x)


# init simulator
sim = AdvancedProjectileSimulator()
sim.add_ground(-1000, 1000, 10, ground_function)

# add projectile
projectile_body = sim.add_projectile((0, 100), (500, 400))

# simulate
positions = sim.simulate(duration=5)

# plot trajectory
sim.plot_trajectory(positions, ground_function, -1000, 1000, 10)

# init simulator
sim = ProjectileSimulator()
sim.add_ground(-1000, 1000, 10, ground_function)

# add projectile
projectile_body = sim.add_projectile((0, 100), (500, 400))

# simulate
positions = sim.simulate(duration=5)

# plot trajectory
sim.plot_trajectory(positions, ground_function, -1000, 1000, 10)
