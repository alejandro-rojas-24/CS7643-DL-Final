# A class for simulating projectiles in a 2D physics environment
# Based off of the paper SimLM: Can Language Models Infer Parameters of Physical Systems?
# https://arxiv.org/abs/2312.14215

import pymunk
import numpy as np
import matplotlib.pyplot as plt
import time
from config import SIM_MAX_DURATION

from abc import ABC, abstractmethod
import pybullet as p
import pybullet_data


class BaseProjectileSimulator(ABC):
    @abstractmethod
    def add_ground(self, x_min, x_max, step_size, y_func):
        pass

    @abstractmethod
    def add_projectile(self, initial_height, initial_velocity):
        pass

    @abstractmethod
    def simulate(self, target_bounces, duration):
        pass


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
        self.ball_collision_type = 1
        self.ground_collision_type = 1
        self._setup_space()
        self.bounce_locations = []
        self.projectile_body = None  # Keep track of the projectile

    def _setup_space(self):
        """Initializes or resets the pymunk space and collision handler."""
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.bounce_locations = []

        # Collision handler to detect bounces
        handler = self.space.add_collision_handler(
            self.ball_collision_type,
            self.ground_collision_type)
        handler.post_solve = self._log_bounce

    def _log_bounce(self, arbiter, space, data):
        """Callback function to record bounce locations."""
        # Ensure the collision involves the projectile
        if self.projectile_body and (
            arbiter.shapes[0].body == self.projectile_body
            or arbiter.shapes[1].body == self.projectile_body
        ):
            # Get the contact point(s)
            contact_point = arbiter.contact_point_set.points[0].point_a
            self.bounce_locations.append(contact_point.x)
            print(f"Debug: Bounce detected at x={contact_point.x:.2f}")

    def add_ground(
        self, x_min, x_max, step_size, y_func, friction=0.8
    ):
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
                (x_vals[i + 1], y_vals[i + 1]),
                radius=0.01,
            )
            segment.friction = friction
            segment.collision_type = self.ground_collision_type
            self.space.add(segment)
        # For plotting later
        self.ground_x_vals = x_vals
        self.ground_y_vals = y_vals
        self.ground_func = y_func

    def add_projectile(
        self,
        initial_height,
        initial_velocity,
        mass=1,
        radius=0.05,
        elasticity=0.9,
    ):
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
            print(
                f"Warning: Initial height {initial_height} is less than radius {radius}. Adjusting to radius."
            )
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
        shape.collision_type = self.ball_collision_type

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
        self.bounce_locations = []  # Reset bounces for this run
        steps = int(duration * self.fps)
        start_time = time.time()

        print(f"Debug: Starting simulation. Target bounces: {target_bounces}")
        print(
            f"Debug: Initial position: {self.projectile_body.position}, velocity: {self.projectile_body.velocity}"
        )

        for step in range(steps):
            self.space.step(self.dt)
            if len(self.bounce_locations) >= target_bounces:
                print(
                    f"Debug: Reached target bounces ({target_bounces}) at step {step}."
                )
                break
            # Check for projectile leaving bounds or stopping
            if (
                self.projectile_body.position.y < -10
            ):  # If it falls far below typical ground
                print(f"Debug: Projectile fell out of bounds at step {step}.")
                break

        end_time = time.time()
        print(
            f"Debug: Simulation finished. Bounces recorded: {len(self.bounce_locations)}. Duration: {end_time - start_time:.2f}s"
        )
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
            positions.append(
                self.projectile_body.position
            )  # Get position of the *actual* body

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
        if hasattr(self, "ground_x_vals") and hasattr(self, "ground_y_vals"):
            plt.plot(self.ground_x_vals, self.ground_y_vals, color="g", label="Ground")
        else:
            print("Warning: Ground data not available for plotting.")

        # Plot trajectory
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], color="r", label="Projectile Path")

        # Plot bounces if available
        if self.bounce_locations:
            bounce_ys = [
                self.ground_func(x) for x in self.bounce_locations
            ]  # Approximate Y on ground
            print(f"Debug: Final bounce locations: {self.bounce_locations}")
            plt.scatter(
                self.bounce_locations, bounce_ys, color="b", zorder=5, label="Bounces"
            )

        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        # Adjust limits dynamically based on trajectory
        if len(positions) > 0:
            x_min_traj, x_max_traj = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min_traj, y_max_traj = np.min(positions[:, 1]), np.max(positions[:, 1])
            plt.xlim(
                min(self.ground_x_vals.min(), x_min_traj - 5),
                max(self.ground_x_vals.max(), x_max_traj + 5),
            )
            plt.ylim(min(self.ground_y_vals.min(), y_min_traj - 5), max(y_max_traj + 5))

        plt.show()


# Advanced Projectile Class with Aerodynamic Drag (2D)
class AdvancedProjectileSimulator(ProjectileSimulator):
    def __init__(
        self, fps=1000, gravity_y=-9.81, drag_coefficient=0.47, air_density=1.225
    ):
        """
        Initialize the advanced projectile simulator with aerodynamic drag.
        Args:
            fps (int): Frames per second for the simulation.
            gravity_y (float): Vertical gravity acceleration (m/s^2).
            drag_coefficient (float): Dimensionless drag coefficient (typical for a sphere is 0.47).
            air_density (float): Density of air in kg/m^3 (sea level is about 1.225).
        """
        super().__init__(fps, gravity_y)
        self.drag_coeff = drag_coefficient
        self.air_density = air_density

    def _apply_drag(self):
        """
        Applies aerodynamic drag force to the projectile using the quadratic drag model.

        The drag force is calculated as:
            F_drag = 0.5 * C_d * rho * v^2 * A
        where:
            C_d = drag coefficient,
            rho = air density,
            v = speed of the projectile,
            A = cross-sectional area (pi * r^2 for a sphere).
        The force is applied in the direction opposite to the velocity.
        """
        if self.projectile_body:
            velocity = self.projectile_body.velocity
            speed = np.linalg.norm(velocity)  # magnitude of the velocity

            if speed > 0:
                radius = self.projectile_body.shapes[0].radius
                area = np.pi * (radius**2)
                drag_mag = 0.5 * self.drag_coeff * self.air_density * speed**2 * area
                drag_force = (
                    -drag_mag * velocity[0] / speed,
                    -drag_mag * velocity[1] / speed,
                )
                self.projectile_body.apply_force_at_world_point(
                    drag_force, self.projectile_body.position
                )  # continuous force to be applied at projectile body position each step

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
        self.bounce_locations = []  # Reset bounces for this run
        steps = int(duration * self.fps)
        start_time = time.time()

        print(f"Debug: Starting simulation. Target bounces: {target_bounces}")
        print(
            f"Debug: Initial position: {self.projectile_body.position}, velocity: {self.projectile_body.velocity}"
        )

        for step in range(steps):
            self._apply_drag()  # Apply drag before physics step
            self.space.step(self.dt)
            if len(self.bounce_locations) >= target_bounces:
                print(
                    f"Debug: Reached target bounces ({target_bounces}) at step {step}."
                )
                break

            # Check for projectile leaving bounds or stopping
            if self.projectile_body.position.y < -10:
                print(f"Debug: Projectile fell out of bounds at step {step}.")
                break

        end_time = time.time()
        print(
            f"Debug: Advanced simulation finished. Bounces recorded: {len(self.bounce_locations)}. Duration: {end_time - start_time:.2f}s"
        )
        print(f"Debug: Final bounce locations: {self.bounce_locations}")

        return self.bounce_locations


class PyBulletProjectileSimulator(BaseProjectileSimulator):
    def __init__(
        self,
        gravity=(0, 0, -9.81),
        timestep=1 / 1000.0,
        gui=True,
        drag_coefficient=0.47,
        air_density=1.225,
    ):
        """
        Initialize 3D projectile simulation environment using PyBullet.

        Args:
            gravity (tuple): 3D gravity vector (x, y, z) in m/s²
            timestep (float): Physics simulation time step in seconds
            gui (bool): Show visual simulation window if True
            drag_coefficient (float): Aerodynamic drag coefficient (dimensionless)
            air_density (float): Air density in kg/m³
        """
        super().__init__()
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity)
        p.setTimeStep(timestep)

        self.ground = None
        self.projectile = None
        self.trajectory = []
        self.bounce_locations = []
        self.drag_coeff = drag_coefficient
        self.air_density = air_density
        self.in_contact = False
        print(
            f"Debug: Simulator initialized | Gravity: {gravity} | Timestep: {timestep}"
        )

    def add_ground(self, terrain_func=None):
        """
        Add ground plane to simulation.

        Args:
            terrain_func (function): Optional heightfield generator function
        """
        if terrain_func:
            print("Debug: Custom terrain function detected.")
        else:
            self.ground = p.loadURDF("plane.urdf")
            print("Debug: Flat ground plane added")

    def add_projectile(
        self,
        initial_position=(0, 0, 0),
        initial_velocity=(0, 0, 0),
        mass=1,
        radius=0.05,
    ):
        """
        Create spherical projectile with specified initial conditions.

        Args:
            initial_position (tuple): 3D launch position (x, y, z) in meters
            initial_velocity (tuple): 3D initial velocity vector (vx, vy, vz) in m/s
            mass (float): Projectile mass in kg
            radius (float): Projectile radius in meters

        Returns:
            int: PyBullet body ID of created projectile
        """
        if self.projectile:
            p.removeBody(self.projectile)
            print("Debug: Removed existing projectile")

        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]
        )

        self.projectile = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=initial_position,
        )
        p.resetBaseVelocity(self.projectile, initial_velocity)
        print(
            f"Debug: Projectile created | Position: {initial_position} | Velocity: {initial_velocity}"
        )
        return self.projectile

    def simulate(self, target_bounces, duration=20):
        """
        Run physics simulation until target bounce count or duration reached.

        Args:
            target_bounces (int): Number of ground impacts to detect
            duration (float): Maximum simulation time in seconds

        Returns:
            list: 3D positions of detected bounce locations
        """
        self.trajectory = []
        self.bounce_locations = []

        steps = int(duration / p.getPhysicsEngineParameters()["fixedTimeStep"])
        start_time = time.time()

        print(f"Debug: Simulation started | Target bounces: {target_bounces}")

        if self.projectile:
            pos, _ = p.getBasePositionAndOrientation(self.projectile)
            vel, _ = p.getBaseVelocity(self.projectile)
            print(f"Debug: Initial state | Position: {pos} | Velocity: {vel}")

        for step in range(steps):
            if self.projectile:
                self._apply_drag()
                pos, _ = p.getBasePositionAndOrientation(self.projectile)
                self.trajectory.append(pos)

                # Contact detection and handling
                contacts = p.getContactPoints(self.projectile)
                if contacts and not self.in_contact:
                    self.bounce_locations.append(pos)
                    print(f"Debug: Bounce detected | Step: {step} | Position: {pos}")
                    self.in_contact = True
                elif not contacts:
                    self.in_contact = False

                # Boundary check
                if pos[2] < -10:
                    print(f"Debug: Projectile out of bounds | Step: {step}")
                    break

                if len(self.bounce_locations) >= target_bounces:
                    print(f"Debug: Target bounces reached | Step: {step}")
                    break

            p.stepSimulation()

        end_time = time.time()
        print(f"Debug: Simulation completed | Duration: {end_time-start_time:.2f}s")
        p.disconnect()
        return self.bounce_locations

    def _apply_drag(self):
        """Apply aerodynamic drag force using quadratic drag model."""
        velocity, _ = p.getBaseVelocity(self.projectile)
        speed = np.linalg.norm(velocity)

        if speed > 0:
            visual_data = p.getVisualShapeData(self.projectile)[0]
            radius = visual_data[3][0]

            area = np.pi * (radius**2)
            drag_force = (
                -0.5 * self.drag_coeff * self.air_density * area * speed * velocity
            )
            p.applyExternalForce(
                self.projectile, -1, drag_force.tolist(), [0, 0, 0], p.WORLD_FRAME
            )
            print(f"Debug: Drag applied | Force: {np.linalg.norm(drag_force):.2f}N")

    def get_trajectory(self):
        """
        Retrieve recorded projectile trajectory.

        Returns:
            ndarray: Nx3 array of (x, y, z) positions
        """
        return np.array(self.trajectory)

    def plot_trajectory(self, title="3D Projectile Trajectory"):
        """
        Visualize trajectory and bounce locations in 3D.

        Args:
            title (str): Plot title text
        """
        traj = self.get_trajectory()
        if len(traj) == 0:
            print("Error: No trajectory data to plot")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Trajectory")

        if self.bounce_locations:
            bounces = np.array(self.bounce_locations)
            ax.scatter(
                bounces[:, 0], bounces[:, 1], bounces[:, 2], c="r", label="Bounces"
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        plt.show()
