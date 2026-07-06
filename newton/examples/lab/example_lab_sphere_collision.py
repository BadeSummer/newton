# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Lab Sphere Collision
#
# A rolling sphere collides with a stationary sphere on a platform.
#
# Command: python -m newton.examples lab_sphere_collision
#
###########################################################################

from argparse import BooleanOptionalAction

import numpy as np
import warp as wp

import newton
import newton.examples


PARAMS = {
    "fps": 60,
    "sim_substeps": 5,
    "num_frames": 260,
    "gravity": -9.81,
    "sphere_radius": 0.05,
    "sphere_density": 1000.0,
    "initial_speed": 1.0,
    "moving_start_x": -0.55,
    "stationary_start_x": 0.0,
    "contact_ke": 1.0e6,
    "contact_kd": 0.0,
    "contact_mu": 0.8,
    "solver_iterations": 12,
    "rigid_contact_hard": True,
    "platform_half_x": 1.0,
    "platform_half_y": 0.35,
    "platform_thickness": 0.04,
    "contact_margin": 0.001,
    "contact_gap": 0.0,
    "history_size": 720,
}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS.copy()
        self.params["sphere_radius"] = args.radius
        self.params["initial_speed"] = args.speed
        self.params["contact_ke"] = args.contact_ke
        self.params["contact_kd"] = args.contact_kd
        self.params["solver_iterations"] = args.solver_iterations
        self.params["rigid_contact_hard"] = args.rigid_contact_hard
        self.report = args.report

        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.radius = self.params["sphere_radius"]
        self.initial_speed = self.params["initial_speed"]

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = self.params["sphere_density"]
        cfg.mu = self.params["contact_mu"]
        cfg.ke = self.params["contact_ke"]
        cfg.kd = self.params["contact_kd"]
        cfg.kf = 0.0
        cfg.margin = self.params["contact_margin"]
        cfg.gap = self.params["contact_gap"]

        platform_cfg = cfg.copy()
        platform_thickness = self.params["platform_thickness"]
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -platform_thickness * 0.5), wp.quat_identity()),
            hx=self.params["platform_half_x"],
            hy=self.params["platform_half_y"],
            hz=platform_thickness * 0.5,
            cfg=platform_cfg,
            color=wp.vec3(0.45, 0.48, 0.50),
            label="platform",
        )

        self.moving_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(self.params["moving_start_x"], 0.0, self.radius),
                wp.quat_identity(),
            ),
            label="moving_sphere",
        )
        builder.add_shape_sphere(
            self.moving_body,
            radius=self.radius,
            cfg=cfg,
            color=wp.vec3(0.22, 0.50, 0.78),
        )

        self.stationary_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(self.params["stationary_start_x"], 0.0, self.radius),
                wp.quat_identity(),
            ),
            label="stationary_sphere",
        )
        builder.add_shape_sphere(
            self.stationary_body,
            radius=self.radius,
            cfg=cfg,
            color=wp.vec3(0.82, 0.30, 0.25),
        )

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            rigid_contact_hard=self.params["rigid_contact_hard"],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        qd = self.state_0.body_qd.numpy()
        qd[self.moving_body, 0] = self.initial_speed
        qd[self.moving_body, 4] = self.initial_speed / self.radius
        self.state_0.body_qd = wp.array(qd, dtype=wp.spatial_vector, device=self.model.device)

        mass = self.model.body_mass.numpy()
        self.sphere_mass = float(mass[self.moving_body])
        self.min_distance = float("inf")
        self.collision_detected = False
        self.moving_speed_x = self.initial_speed
        self.stationary_speed_x = 0.0
        self.total_momentum_x = self.sphere_mass * self.initial_speed
        self.speed_history = [self.moving_speed_x]
        self.stationary_speed_history = [self.stationary_speed_x]

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.4, -1.1, 0.55), -18.0, 135.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_monitor, position="free")

        self.graph = None

    def _sample(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        moving_pos = body_q[self.moving_body, :3]
        stationary_pos = body_q[self.stationary_body, :3]
        distance = float(np.linalg.norm(stationary_pos - moving_pos))
        self.min_distance = min(self.min_distance, distance)
        if distance <= 2.1 * self.radius:
            self.collision_detected = True

        self.moving_speed_x = float(body_qd[self.moving_body, 0])
        self.stationary_speed_x = float(body_qd[self.stationary_body, 0])
        self.total_momentum_x = self.sphere_mass * (self.moving_speed_x + self.stationary_speed_x)

        self.speed_history.append(self.moving_speed_x)
        self.stationary_speed_history.append(self.stationary_speed_x)
        if len(self.speed_history) > self.params["history_size"]:
            self.speed_history.pop(0)
            self.stationary_speed_history.pop(0)

    def render_monitor(self, imgui):
        ui = getattr(self.viewer, "ui", None)
        if ui is None or not getattr(ui, "is_available", False):
            return

        io = ui.io
        window_width = 460
        window_height = 520
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_width - 16, 96),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))

        flags = imgui.WindowFlags_.no_resize.value
        if imgui.begin("Lab Monitor", flags=flags):
            imgui.text("Sphere Collision")
            imgui.separator()
            imgui.text(f"Time: {self.sim_time:.3f} s")
            imgui.text(f"Sphere mass: {self.sphere_mass:.6f} kg")
            imgui.text(f"Moving vx: {self.moving_speed_x:.6f} m/s")
            imgui.text(f"Stationary vx: {self.stationary_speed_x:.6f} m/s")
            imgui.text(f"Total px: {self.total_momentum_x:.6f} kg*m/s")
            imgui.text(f"Min center distance: {self.min_distance:.6f} m")
            imgui.text(f"Collision detected: {self.collision_detected}")
            imgui.separator()

            graph_size = imgui.ImVec2(-1, 135)
            moving_history = np.array(self.speed_history, dtype=np.float32)
            stationary_history = np.array(self.stationary_speed_history, dtype=np.float32)
            speed_limit = max(abs(self.initial_speed), abs(self.moving_speed_x), abs(self.stationary_speed_x), 0.1) * 1.2
            imgui.plot_lines(
                "Moving vx",
                moving_history,
                graph_size=graph_size,
                overlay_text=f"{self.moving_speed_x:.4f} m/s",
                scale_min=-speed_limit,
                scale_max=speed_limit,
            )
            imgui.plot_lines(
                "Stationary vx",
                stationary_history,
                graph_size=graph_size,
                overlay_text=f"{self.stationary_speed_x:.4f} m/s",
                scale_min=-speed_limit,
                scale_max=speed_limit,
            )
            imgui.separator()
            imgui.text("Solver: VBD")
            imgui.text(f"Hard contacts: {self.params['rigid_contact_hard']}")
            imgui.text(f"Contact ke: {self.params['contact_ke']:.3e}")
            imgui.text(f"Contact kd: {self.params['contact_kd']:.3e}")
        imgui.end()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._sample()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Sphere collision lab produced non-finite body transforms")
        if not self.collision_detected:
            raise ValueError("Sphere collision lab did not detect sphere contact")
        if self.stationary_speed_x <= 0.05:
            raise ValueError(f"Stationary sphere did not gain enough speed: vx={self.stationary_speed_x:.6f} m/s")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        parser.add_argument("--radius", type=float, default=PARAMS["sphere_radius"], help="Sphere radius [m].")
        parser.add_argument("--speed", type=float, default=PARAMS["initial_speed"], help="Initial rolling speed [m/s].")
        parser.add_argument("--contact-ke", type=float, default=PARAMS["contact_ke"], help="Contact stiffness [N/m].")
        parser.add_argument("--contact-kd", type=float, default=PARAMS["contact_kd"], help="Contact damping [N*s/m].")
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=PARAMS["solver_iterations"],
            help="VBD solver iterations per substep.",
        )
        parser.add_argument(
            "--rigid-contact-hard",
            action=BooleanOptionalAction,
            default=PARAMS["rigid_contact_hard"],
            help="Use VBD hard rigid contacts instead of soft penalty contacts.",
        )
        parser.add_argument("--report", action="store_true", help="Reserved for lab consistency.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
