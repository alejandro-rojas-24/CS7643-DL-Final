import pymunk
import numpy as np
import matplotlib.pyplot as plt

FPS = 1000  # frames per second to ensure high precision and deterministic simulation
DT = 1 / FPS  # duration of each simulation step
SPACE_GRAVITY = (0, -981)  # (cm/s^2)

# init space
space = pymunk.Space()
space.gravity = SPACE_GRAVITY

def rolling_ground_function(x):
    return 100 * np.sin(0.01 * x)

# Generate piecewise linear ground geometry
x_min, x_max, step = -1000, 1000, 10
x_vals = np.arange(x_min, x_max + step, step)
y_vals = rolling_ground_function(x_vals)

# Create ground as static segments in Pymunk
for i in range(len(x_vals) - 1):
    segment = pymunk.Segment(space.static_body,
                             (x_vals[i], y_vals[i]),
                             (x_vals[i+1], y_vals[i+1]),
                             radius=1)
    segment.friction = 0.8  # friction coefficient of the ground
    space.add(segment)

# Add projectile (represented as a rigid-body circle)
mass = 1  # mass of the projectile
radius = 5  # radius of the projectile
moment = pymunk.moment_for_circle(mass, 0, radius)  # moment of inertia for a circle
projectile_body = pymunk.Body(mass, moment)
projectile_body.position = (-900, 300)  # Initial position of the projectile
projectile_body.velocity = (500, 400)   # Initial velocity of the projectile

# Define the shape and physical properties of the projectile
projectile_shape = pymunk.Circle(projectile_body, radius)
projectile_shape.elasticity = 0.8  # elasticity (bounciness)
projectile_shape.friction = 0.5  # friction coefficient
space.add(projectile_body, projectile_shape)

# Simulation loop: run the simulation and record positions
positions = []
sim_duration = 5  # total simulation duration in seconds
steps = int(sim_duration * FPS)

for _ in range(steps):
    space.step(DT)  # advance the simulation by DT seconds
    positions.append(projectile_body.position)

# Plot the trajectory of the projectile along with the ground geometry
positions = np.array(positions)
plt.plot(x_vals, y_vals, 'g', label="Ground")
plt.plot(positions[:, 0], positions[:, 1], 'r', label="Projectile Path")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.title("Projectile Trajectory using Pymunk")
plt.show()