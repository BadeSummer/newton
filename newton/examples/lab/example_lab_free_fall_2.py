# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Lab Free Fall 2
#
# Drops a sphere onto a freely moving rectangular block.
#
# Command: python -m newton.examples lab_free_fall_2
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
    "drop_height": 1.0,
    "sphere_initial_vz": 0.0,
    "sphere_radius": 0.05,
    "sphere_density": 1000.0,
    "block_half_extents": (0.25, 0.18, 0.04),
    "block_density": 1000.0,
    "block_initial_height": 0.04,
    "platform_half_extent": 0.75,
    "platform_thickness": 0.04,
    "contact_ke": 3e5,
    "contact_kd": 0.0,
    "contact_mu": 0.0,
    "solver_iterations": 10,
    "rigid_contact_hard": True,
    "num_frames": 300,
    "gravity": -9.81,
    "contact_margin": 0.001,
    "contact_gap": 0.0,
    "history_size": 720,
}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS.copy()
        self.params["drop_height"] = args.height
        self.params["sphere_radius"] = args.radius
        self.params["contact_ke"] = args.contact_ke
        self.params["contact_kd"] = args.contact_kd
        self.params["solver_iterations"] = args.solver_iterations
        self.params["rigid_contact_hard"] = args.rigid_contact_hard
        self.params["sphere_initial_vz"] = args.sphere_initial_vz
        self.report = args.report

        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.radius = self.params["sphere_radius"]
        hx, hy, hz = self.params["block_half_extents"]
        self.block_half_extents = (hx, hy, hz)
        self.block_initial_z = self.params["block_initial_height"]
        self.sphere_initial_z = self.block_initial_z + hz + self.radius + self.params["drop_height"]

        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = self.params["sphere_density"]
        sphere_cfg.mu = self.params["contact_mu"]
        sphere_cfg.ke = self.params["contact_ke"]
        sphere_cfg.kd = self.params["contact_kd"]
        sphere_cfg.kf = 0.0
        sphere_cfg.margin = self.params["contact_margin"]
        sphere_cfg.gap = self.params["contact_gap"]

        block_cfg = newton.ModelBuilder.ShapeConfig()
        block_cfg.density = self.params["block_density"]
        block_cfg.mu = self.params["contact_mu"]
        block_cfg.ke = self.params["contact_ke"]
        block_cfg.kd = self.params["contact_kd"]
        block_cfg.kf = 0.0
        block_cfg.margin = self.params["contact_margin"]
        block_cfg.gap = self.params["contact_gap"]

        platform_cfg = block_cfg.copy()
        platform_cfg.density = 0.0

        platform_thickness = self.params["platform_thickness"]
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -platform_thickness * 0.5), wp.quat_identity()),
            hx=self.params["platform_half_extent"],
            hy=self.params["platform_half_extent"],
            hz=platform_thickness * 0.5,
            cfg=platform_cfg,
            color=wp.vec3(0.45, 0.48, 0.50),
            label="platform",
        )

        self.block_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.block_initial_z), wp.quat_identity()),
            label="free_block",
        )
        builder.add_shape_box(
            self.block_body,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=block_cfg,
            color=wp.vec3(0.82, 0.30, 0.25),
        )

        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.sphere_initial_z), wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(
            self.sphere_body,
            radius=self.radius,
            cfg=sphere_cfg,
            color=wp.vec3(0.22, 0.50, 0.78),
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
        qd[self.sphere_body, 2] = self.params["sphere_initial_vz"]
        self.state_0.body_qd = wp.array(qd, dtype=wp.spatial_vector, device=self.model.device)

        masses = self.model.body_mass.numpy()
        self.sphere_mass = float(masses[self.sphere_body])
        self.block_mass = float(masses[self.block_body])
        self.sphere_height = self.sphere_initial_z - self.radius
        self.block_top_height = self.block_initial_z + hz
        self.sphere_vz = 0.0
        self.block_vz = 0.0
        self.min_gap = float("inf")
        self.contact_detected = False
        self.max_block_upward_vz = 0.0
        self.max_block_speed_delta = 0.0
        self.sphere_height_history = [self.sphere_height]
        self.block_top_history = [self.block_top_height]

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.1, -1.1, 0.85), -18.0, 135.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_monitor, position="free")

        self.graph = None

    def _sample(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        sphere_center_z = float(body_q[self.sphere_body, 2])
        block_center_z = float(body_q[self.block_body, 2])
        self.sphere_height = sphere_center_z - self.radius
        self.block_top_height = block_center_z + self.block_half_extents[2]
        self.sphere_vz = float(body_qd[self.sphere_body, 2])
        self.block_vz = float(body_qd[self.block_body, 2])
        self.max_block_upward_vz = max(self.max_block_upward_vz, self.block_vz)
        self.max_block_speed_delta = max(self.max_block_speed_delta, abs(self.block_vz))
        gap = self.sphere_height - self.block_top_height
        self.min_gap = min(self.min_gap, gap)
        if gap <= 2.0 * self.params["contact_margin"]:
            self.contact_detected = True

        self.sphere_height_history.append(self.sphere_height)
        self.block_top_history.append(self.block_top_height)
        if len(self.sphere_height_history) > self.params["history_size"]:
            self.sphere_height_history.pop(0)
            self.block_top_history.pop(0)

    def render_monitor(self, imgui):
        ui = getattr(self.viewer, "ui", None)
        if ui is None or not getattr(ui, "is_available", False):
            return

        io = ui.io
        window_width = 460
        window_height = 560
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_width - 16, 96),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))

        flags = imgui.WindowFlags_.no_resize.value
        if imgui.begin("Lab Monitor", flags=flags):
            imgui.text("Free Fall 2")
            imgui.separator()
            imgui.text(f"Time: {self.sim_time:.3f} s")
            imgui.text(f"Sphere mass: {self.sphere_mass:.6f} kg")
            imgui.text(f"Block mass: {self.block_mass:.6f} kg")
            imgui.text(f"Sphere bottom: {self.sphere_height:.6f} m")
            imgui.text(f"Block top: {self.block_top_height:.6f} m")
            imgui.text(f"Gap: {self.sphere_height - self.block_top_height:.6f} m")
            imgui.text(f"Sphere vz: {self.sphere_vz:.6f} m/s")
            imgui.text(f"Block vz: {self.block_vz:.6f} m/s")
            imgui.text(f"Max block upward vz: {self.max_block_upward_vz:.6f} m/s")
            imgui.text(f"Contact detected: {self.contact_detected}")
            imgui.separator()

            graph_size = imgui.ImVec2(-1, 140)
            sphere_history = np.array(self.sphere_height_history, dtype=np.float32)
            block_history = np.array(self.block_top_history, dtype=np.float32)
            height_max = max(max(self.sphere_height_history), max(self.block_top_history), self.sphere_initial_z) * 1.05
            height_min = min(min(self.sphere_height_history), min(self.block_top_history), 0.0)
            imgui.plot_lines(
                "Sphere bottom",
                sphere_history,
                graph_size=graph_size,
                overlay_text=f"{self.sphere_height:.4f} m",
                scale_min=height_min,
                scale_max=height_max,
            )
            imgui.plot_lines(
                "Block top",
                block_history,
                graph_size=graph_size,
                overlay_text=f"{self.block_top_height:.4f} m",
                scale_min=height_min,
                scale_max=height_max,
            )
            imgui.separator()
            imgui.text("Solver: VBD")
            imgui.text(f"Hard contacts: {self.params['rigid_contact_hard']}")
            imgui.text(f"Contact ke: {self.params['contact_ke']:.3e}")
            imgui.text(f"Contact kd: {self.params['contact_kd']:.3e}")
            imgui.text(f"Initial sphere vz: {self.params['sphere_initial_vz']:.3f} m/s")
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
            raise ValueError("Free-fall 2 lab produced non-finite body transforms")
        if not self.contact_detected:
            raise ValueError("Free-fall 2 lab did not detect sphere/block contact")
        if self.max_block_speed_delta < 0.05:
            raise ValueError(f"Block did not respond enough to impact: max |vz|={self.max_block_speed_delta:.6f}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        parser.add_argument("--height", type=float, default=PARAMS["drop_height"], help="Sphere drop height [m].")
        parser.add_argument("--radius", type=float, default=PARAMS["sphere_radius"], help="Sphere radius [m].")
        parser.add_argument("--contact-ke", type=float, default=PARAMS["contact_ke"], help="Contact stiffness [N/m].")
        parser.add_argument("--contact-kd", type=float, default=PARAMS["contact_kd"], help="Contact damping [N*s/m].")
        parser.add_argument(
            "--sphere-initial-vz",
            type=float,
            default=PARAMS["sphere_initial_vz"],
            help="Initial vertical velocity of the sphere [m/s].",
        )
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
