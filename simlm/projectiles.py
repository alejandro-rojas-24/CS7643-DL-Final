# A class for simulating projectiles in a 2D physics environment
# Based off of the paper SimLM: Can Language Models Infer Parameters of Physical Systems?
# https://arxiv.org/abs/2312.14215

import pymunk
import numpy as np
import matplotlib.pyplot as plt
import time
from config import ( SIM_MAX_DURATION)

class ProjectileSimulator:
    def __init__(self, fps=1000, gravity_y=-9.81):
        """
        Initialize the projectile simulator environment

        Args:
            fps (int): Frames per second to simulate the physics
            space_gravity (tuple): Gravity vector (x, y) in cm/s^2
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.gravity = (0, gravity_y)
        self._setup_space()
        self.bounce_locations = []
        self.projectile_body = None # Keep track of the projectile

    def _setup_space(self):
        """Initializes or resets the pymunk space and collision handler."""
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.bounce_locations = [] 

        # Collision handler to detect bounces
        handler = self.space.add_collision_handler(0, 0)
        handler.post_solve = self._log_bounce

    def _log_bounce(self, arbiter, space, data):
        """Callback function to record bounce locations."""
        # Ensure the collision involves the projectile
        if self.projectile_body and (arbiter.shapes[0].body == self.projectile_body or arbiter.shapes[1].body == self.projectile_body):
            # Get the contact point(s)
            contact_point = arbiter.contact_point_set.points[0].point_a
            self.bounce_locations.append(contact_point.x)
            print(f"Debug: Bounce detected at x={contact_point.x:.2f}")

    def add_ground(self, x_min, x_max, step_size, y_func, friction=0.8, collision_type=0):
        """
        Setup ground geometry for the simulation

        Args:
            x_min (float): Min x-coordinate of the ground (meters).
            x_max (float): Max x-coordinate of the ground (meters).
            step_size (float): Step size for creating ground segments (meters).
            y_func (function): Function defining y-coordinate from x.
            friction (float): Friction coefficient of the ground.
            collision_type (int): Pymunk collision type for the ground segments.
        """
        x_vals = np.arange(x_min, x_max + step_size, step_size)
        y_vals = y_func(x_vals)

        for i in range(len(x_vals) - 1):
            segment = pymunk.Segment(
                self.space.static_body,
                (x_vals[i], y_vals[i]),
                (x_vals[i+1], y_vals[i+1]),
                radius=0.01
            )
            segment.friction = friction
            segment.collision_type = collision_type
            # segment.filter = pymunk.ShapeFilter(group=1)
            self.space.add(segment)
        # For plotting later
        self.ground_x_vals = x_vals
        self.ground_y_vals = y_vals
        self.ground_func = y_func


    def add_projectile(self, initial_height, initial_velocity,
                       mass=1, radius=0.05,
                       elasticity=0.9, collision_type=0):
        """
        Adds the projectile based on the paper's parameters.

        Args:
            initial_height (float): Initial height (y-coordinate) in meters.
            initial_velocity (float): Initial horizontal velocity (vx) in m/s.
            mass (float): Mass of the projectile in g
            radius (float): Radius of the projectile in cm
            elasticity (float): Elasticity of the projectile
            collision_type (int): Pymunk collision type for the projectile.

        Returns:
            pymunk.Body: The created projectile body.
        """
        if initial_height < radius:
             print(f"Warning: Initial height {initial_height} is less than radius {radius}. Adjusting to radius.")
             initial_height = radius

        # create projectile body
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.projectile_body = pymunk.Body(mass, moment)
        # Paper implies launch from x=0
        self.projectile_body.position = (0, initial_height)
        # Paper implies v is horizontal velocity, initial vertical velocity is 0
        self.projectile_body.velocity = (initial_velocity, 0)

        shape = pymunk.Circle(self.projectile_body, radius)
        shape.elasticity = elasticity
        shape.collision_type = collision_type

        # Ensure projectile can collide with ground
        shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(self.projectile_body, shape)

        return self.projectile_body

    def simulate(self, target_bounces, duration=20):
        """
        Run simulation for a give duration or reaches target number of bounces

        Args:
            target_bounces (int): The number of bounces to simulate
            duration (float): Duration of the simulation in seconds

        Returns:
            list: A list of x-coordinates (in meters) for each detected bounce.
                  Returns fewer bounces if max_duration is reached first.
        """
        self.bounce_locations = [] # Reset bounces for this run
        steps = int(duration * self.fps)
        start_time = time.time()

        print(f"Debug: Starting simulation. Target bounces: {target_bounces}") 
        print(f"Debug: Initial position: {self.projectile_body.position}, velocity: {self.projectile_body.velocity}")

        for step in range(steps):
            self.space.step(self.dt)
            if len(self.bounce_locations) >= target_bounces:
                print(f"Debug: Reached target bounces ({target_bounces}) at step {step}.")
                break
            # Check for projectile leaving bounds or stopping
            if self.projectile_body.position.y < -10: # If it falls far below typical ground
                 print(f"Debug: Projectile fell out of bounds at step {step}.")
                 break

        end_time = time.time()
        print(f"Debug: Simulation finished. Bounces recorded: {len(self.bounce_locations)}. Duration: {end_time - start_time:.2f}s") 
        print(f"Debug: Final bounce locations: {self.bounce_locations}")

        return self.bounce_locations

    def get_trajectory(self, duration):
        """
        Simulates and returns the full trajectory for plotting/debugging.
        Note: This runs a new simulation from the current state if called after simulate.
              It's better to run this instead of simulate if you need the path.
              We could modify simulate to also store trajectory points.
        """
        # Re-setup space if needed, or ensure state is correct
        # For simplicity, let's assume this is called on a freshly setup space
        # This part needs careful state management if used interchangeably with run_simulation
        if not self.projectile_body:
            print("Error: Add projectile before getting trajectory.")
            return []

        initial_pos = self.projectile_body.position
        initial_vel = self.projectile_body.velocity

        # Temporarily store current bounces and reset
        original_bounces = list(self.bounce_locations)
        self.bounce_locations = []

        steps = int(duration * self.fps)
        positions = []
        current_pos = initial_pos
        current_vel = initial_vel

        for _ in range(steps):
            self.space.step(self.dt)
            positions.append(self.projectile_body.position) # Get position of the *actual* body

        # Restore original state (optional, depends on use case)
        self.projectile_body.position = initial_pos
        self.projectile_body.velocity = initial_vel
        self.bounce_locations = original_bounces

        return np.array(positions)

    def plot_trajectory(self, positions, title="Projectile Trajectory"):
        """Plots a recorded trajectory and the ground."""
        if positions is None or len(positions) == 0:
            print("No trajectory data to plot.")
            return

        plt.figure(figsize=(10, 6))
        # Plot ground
        if hasattr(self, 'ground_x_vals') and hasattr(self, 'ground_y_vals'):
            plt.plot(self.ground_x_vals, self.ground_y_vals, color='g', label='Ground')
        else:
            print("Warning: Ground data not available for plotting.")

        # Plot trajectory
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], color='r', label='Projectile Path')

        # Plot bounces if available
        if self.bounce_locations:
            bounce_ys = [self.ground_func(x) for x in self.bounce_locations] # Approximate Y on ground
            plt.scatter(self.bounce_locations, bounce_ys, color='b', zorder=5, label='Bounces')

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis('equal') 

        # Adjust limits dynamically based on trajectory
        if len(positions) > 0:
            x_min_traj, x_max_traj = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min_traj, y_max_traj = np.min(positions[:, 1]), np.max(positions[:, 1])
            plt.xlim(min(self.ground_x_vals.min(), x_min_traj - 5), max(self.ground_x_vals.max(), x_max_traj + 5))
            plt.ylim(min(self.ground_y_vals.min(), y_min_traj - 5), max(y_max_traj + 5))

        plt.show()