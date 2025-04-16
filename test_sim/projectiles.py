# projectiles.py
# A class for simulating projectiles in a 2D physics environment
# Based off of the paper SimLM: Can Language Models Infer Parameters of Physical Systems?
# https://arxiv.org/abs/2312.14215

import pymunk
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


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


class PyBulletProjectile3D:
    def __init__(self, gravity=(0, 0, -9.81), timestep=1 / 1000.0):
        """
        Initialize the PyBullet physics environment with specified parameters.

        Args:
            gravity: 3D vector (x,y,z) for gravity in m/s² - default is Earth gravity in -Z direction
            timestep: Physics simulation timestep in seconds - smaller values give more accurate physics
        """

        # Initialize in GUI mode, could possibly use p.DIRECT if not necessary
        self.client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity)
        p.setTimeStep(timestep)

        # flat ground plane
        self.ground = p.loadURDF("plane.urdf")
        self.projectiles = []
        self.trajectories = []

    def add_projectile(self, position, velocity, mass=1, radius=0.05):
        """
        Create and add a spherical projectile to the simulation.

        Args:
            position: Initial 3D position (x,y,z) in meters
            velocity: Initial 3D velocity (vx,vy,vz) in m/s
            mass: Mass of the projectile in kg
            radius: Radius of the projectile in meters

        Returns:
            The PyBullet body ID of the created projectile
        """

        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]
        )
        projectile = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
        )
        p.resetBaseVelocity(projectile, velocity)
        self.projectiles.append(projectile)
        self.trajectories.append([])

        return projectile

    def _apply_drag(self, body, drag_coeff=0.47, air_density=1.225):
        """
        Apply aerodynamic drag force to a projectile.

        Args:
            body: PyBullet body ID to apply drag to
            drag_coeff: Drag coefficient (0.47 is typical for a sphere)
            air_density: Air density in kg/m³ (1.225 is sea level standard)
        """

        velocity, _ = p.getBaseVelocity(body)
        speed = np.linalg.norm(velocity)
        if speed > 0:
            # Extract radius correctly from visual shape data
            visual_data = p.getVisualShapeData(body)[0]
            dimensions = visual_data[3]
            radius = dimensions[0]

            # Apply standard drag equation: F_drag = 0.5 * ρ * v² * C_d * A
            area = np.pi * (radius**2)
            drag_mag = 0.5 * drag_coeff * air_density * area * (speed**2)
            drag_force = [-drag_mag * v / speed for v in velocity]

            p.applyExternalForce(body, -1, drag_force, [0, 0, 0], p.WORLD_FRAME)

    def simulate(self, duration, apply_drag=True):
        """
        Run the physics simulation for a specified duration.

        Args:
            duration: Time to simulate in seconds
            apply_drag: Whether to apply air resistance (True by default)
        """

        steps = int(duration / p.getPhysicsEngineParameters()["fixedTimeStep"])
        for _ in range(steps):
            if apply_drag:
                for projectile in self.projectiles:
                    self._apply_drag(projectile)
            p.stepSimulation()

            # Record trajectories
            for i, projectile in enumerate(self.projectiles):
                pos, _ = p.getBasePositionAndOrientation(projectile)
                self.trajectories[i].append(pos)

    def plot_trajectory_3d(self):
        """
        Create a 3D plot showing the trajectory of all projectiles.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for trajectory in self.trajectories:
            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]
            z = [p[2] for p in trajectory]
            ax.plot(x, y, z, label="Projectile")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        plt.title("3D Projectile Trajectory")
        plt.show()
