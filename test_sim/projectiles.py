# projectiles.py
# A class for simulating projectiles in a 2D physics environment
# Based off of the paper SimLM: Can Language Models Infer Parameters of Physical Systems?
# https://arxiv.org/abs/2312.14215

import pymunk
import numpy as np
import matplotlib.pyplot as plt


class ProjectileSimulator:
    def __init__(self, fps=1000, gravity=(0, -981)):
        """
        Initialize the projectile simulator environment

        Args:
            fps (int): Frames per second to simulate the physics
            space_gravity (tuple): Gravity vector (x, y) in cm/s^2
        """

        self.fps = fps
        self.dt = 1 / fps
        self.space = pymunk.Space()
        self.space.gravity = gravity

    def add_ground(self, x_min, x_max, step_size, y_func, friction=0.8):
        """
        Setup ground geometry for the simulation

        Args:
            x_min (float): Min x-coordinate of the ground
            x_max (float): Max x-coordinate of the ground
            step_size (float): Step size for the ground
            y_func (function): Function that defines the y-coordinate of the ground as a function of x
        """

        x_vals = np.arange(x_min, x_max + step_size, step_size)
        y_vals = y_func(x_vals)

        for i in range(len(x_vals) - 1):
            segment = pymunk.Segment(
                self.space.static_body,
                (x_vals[i], y_vals[i]),
                (x_vals[i + 1], y_vals[i + 1]),
                radius=1,
            )
            segment.friction = friction
            self.space.add(segment)

    def add_projectile(self, position, velocity, mass=1, radius=5, elasticity=0.9):
        """
        Add projectile to the simulation

        Args:
            position (tuple): Initial position of the projectile (x, y) in cm
            velocity (tuple): Initial velocity of the projectile (vx, vy) in cm/s
            mass (float): Mass of the projectile in g
            radius (float): Radius of the projectile in cm
            elasticity (float): Elasticity of the projectile
        """

        # create projectile body
        moment = pymunk.moment_for_circle(mass, 0, radius)
        projectile_body = pymunk.Body(mass, moment)
        projectile_body.position = position
        projectile_body.velocity = velocity

        shape = pymunk.Circle(projectile_body, radius)
        shape.elasticity = elasticity

        self.space.add(projectile_body, shape)

        return projectile_body

    def simulate(self, duration):
        """
        Run simulation for a given duration

        Args:
            duration (float): Duration of the simulation in seconds
        """

        steps = int(duration * self.fps)
        positions = []

        for _ in range(steps):
            self.space.step(self.dt)
            positions.append(
                [(body.position.x, body.position.y) for body in self.space.bodies]
            )

        return positions

    def plot_trajectory(self, positions, ground_func, x_min, x_max, step_size):
        """
        Plot the trajectory of the projectile

        Args:
            positions (list): List of positions of the projectile
            ground_func (function): Function that defines the y-coordinate of the ground as a function of x
            x_min (float): Min x-coordinate of the ground
            x_max (float): Max x-coordinate of the ground
            step_size (float): Step size for the ground
        """

        x_vals = np.arange(x_min, x_max + step_size, step_size)
        y_vals = ground_func(x_vals)

        plt.plot(x_vals, y_vals, color="g", label="Ground")

        for idx, pos in enumerate(zip(*positions)):
            pos = np.array(pos)
            plt.plot(
                pos[:, 0], pos[:, 1], color="r", label="Projectile" if idx == 0 else ""
            )

        plt.xlabel("X position (cm)")
        plt.ylabel("Y position (cm)")
        plt.legend()
        plt.title("Projectile Trajectory")
        plt.show()


class AdvancedProjectileSimulator(ProjectileSimulator):
    def __init__(self, air_density=0.001225, drag_coeff=0.47, **kwargs):
        """
        Args:
            air_density (float): g/cm³ (1.225 kg/m³ = 0.001225 g/cm³)
            drag_coeff (float): Unitless drag coefficient
        """
        super().__init__(**kwargs)
        self.drag_coeff = drag_coeff
        self.air_density = air_density

    def _apply_drag(self, body):
        """Apply drag force in cm-based units"""
        velocity_cmps = body.velocity
        speed = velocity_cmps.length

        if speed > 0 and body.shapes:
            shape = list(body.shapes)[0]
            radius_cm = shape.radius
            area_cm2 = np.pi * (radius_cm**2)

            # Drag force in g·cm/s²
            drag_mag = 0.5 * self.drag_coeff * self.air_density * speed**2 * area_cm2
            drag_force = (
                -drag_mag * velocity_cmps.x / speed,
                -drag_mag * velocity_cmps.y / speed,
            )

            body.apply_force_at_world_point(drag_force, body.position)

    def simulate(self, duration):
        steps = int(duration * self.fps)
        positions = []

        for _ in range(steps):
            for body in self.space.bodies:
                self._apply_drag(body)
            self.space.step(self.dt)
            positions.append(
                [(body.position.x, body.position.y) for body in self.space.bodies]
            )

        return positions
