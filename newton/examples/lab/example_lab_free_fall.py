# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Lab Free Fall
#
# Drops a sphere from height H onto a platform and records rebound heights.
#
# Command: python -m newton.examples lab_free_fall
#
###########################################################################

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import CollisionPipeline
import newton.examples


PARAMS = {
    "fps": 60,
    "sim_substeps": 5,
    "solver_iterations": 10,
    "drop_height": 1.0,
    "sphere_radius": 0.05,
    "contact_ke": 3000,
    "contact_kd": 0.0,
    "rigid_contact_hard": True,
    "rigid_contact_history": False,
    "rigid_avbd_alpha": 0.0,
    "min_rebound_height": 0.005,
    "num_frames": 360,
    "gravity": -9.81,
    "platform_half_extent": 0.75,
    "platform_thickness": 1.0,
    "contact_margin": 0.0,
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
        self.params["min_rebound_height"] = args.min_rebound_height

        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.drop_height = self.params["drop_height"]
        self.radius = self.params["sphere_radius"]
        self.report = args.report

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = 0.0
        cfg.ke = self.params["contact_ke"]
        cfg.kd = self.params["contact_kd"]
        cfg.kf = 0.0
        cfg.margin = self.params["contact_margin"]
        cfg.gap = self.params["contact_gap"]

        platform_thickness = self.params["platform_thickness"]
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -platform_thickness * 0.5), wp.quat_identity()),
            hx=self.params["platform_half_extent"],
            hy=self.params["platform_half_extent"],
            hz=platform_thickness * 0.5,
            cfg=cfg,
            color=wp.vec3(0.45, 0.48, 0.50),
            label="platform",
        )

        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.radius + self.drop_height), wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(
            self.sphere_body,
            radius=self.radius,
            cfg=cfg,
            color=wp.vec3(0.22, 0.50, 0.78),
        )
        builder.color()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            rigid_contact_hard=self.params["rigid_contact_hard"],
            rigid_contact_history=self.params["rigid_contact_history"],
            rigid_avbd_alpha=self.params["rigid_avbd_alpha"],
        )
        self.collision_pipeline = CollisionPipeline(self.model, contact_matching="latest")
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.prev_height = self.drop_height
        self.prev_velocity = 0.0
        self.current_height = self.drop_height
        self.current_velocity = 0.0
        self.max_height = self.drop_height
        self.height_history = [self.drop_height]
        self.seen_impact = False
        self.bounce_peaks = []

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.2, -2.2, 1.3), -18.0, 135.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_monitor, position="free")

        self.graph = None

    def _sample_height(self):
        center_z = float(self.state_0.body_q.numpy()[self.sphere_body, 2])
        height = center_z - self.radius
        velocity = (height - self.prev_height) / self.sim_dt
        self.current_height = height
        self.current_velocity = velocity
        self.max_height = max(self.max_height, height)
        self.height_history.append(height)
        if len(self.height_history) > self.params["history_size"]:
            self.height_history.pop(0)

        if not self.seen_impact and self.prev_velocity < 0.0 <= velocity:
            self.seen_impact = True

        if self.seen_impact and self.prev_velocity > 0.0 >= velocity:
            peak = max(self.prev_height, height)
            if peak >= self.params["min_rebound_height"]:
                self.bounce_peaks.append(peak)
                if self.report:
                    print(f"bounce {len(self.bounce_peaks)} height: {peak:.6f} m")
            self.seen_impact = False

        self.prev_height = height
        self.prev_velocity = velocity

    def render_monitor(self, imgui):
        ui = getattr(self.viewer, "ui", None)
        if ui is None or not getattr(ui, "is_available", False):
            return

        io = ui.io
        window_width = 420
        window_height = 520
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_width - 16, 96),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))

        flags = imgui.WindowFlags_.no_resize.value
        if imgui.begin("Lab Monitor", flags=flags):
            imgui.text("Free Fall")
            imgui.separator()
            imgui.text(f"Time: {self.sim_time:.3f} s")
            imgui.text(f"Height: {self.current_height:.6f} m")
            imgui.text(f"Vertical velocity: {self.current_velocity:.6f} m/s")
            imgui.text(f"Maximum height: {self.max_height:.6f} m")
            imgui.separator()
            graph_size = imgui.ImVec2(-1, 150)
            height_history = np.array(self.height_history, dtype=np.float32)
            imgui.plot_lines(
                "Height",
                height_history,
                graph_size=graph_size,
                overlay_text=f"{self.current_height:.4f} m",
                scale_min=0.0,
                scale_max=max(self.drop_height, self.max_height, self.radius) * 1.05,
            )
            imgui.separator()
            imgui.text(f"Drop height H: {self.drop_height:.6f} m")
            imgui.text("Solver: VBD")
            imgui.text(f"Hard contacts: {self.params['rigid_contact_hard']}")
            imgui.text(f"Contact ke: {self.params['contact_ke']:.3e}")
            imgui.text(f"Contact kd: {self.params['contact_kd']:.3e}")
            imgui.text(f"Min rebound: {self.params['min_rebound_height']:.6f} m")
            if self.bounce_peaks:
                imgui.separator()
                imgui.text("Rebound peaks")
                for i, peak in enumerate(self.bounce_peaks[-5:], start=max(1, len(self.bounce_peaks) - 4)):
                    ratio = peak / self.drop_height if self.drop_height > 0.0 else 0.0
                    imgui.text(f"{i}: {peak:.6f} m ({ratio:.3f} H)")
            else:
                imgui.separator()
                imgui.text("Rebound peaks: none")
        imgui.end()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._sample_height()

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
            raise ValueError("Free-fall lab produced non-finite body transforms")
        if not self.bounce_peaks:
            raise ValueError("Free-fall lab did not record any rebound peak")
        measured = self.bounce_peaks[0]
        if measured <= 0.0 or measured > self.drop_height * 1.25:
            raise ValueError(f"Unexpected rebound height: measured={measured:.6f} m, H={self.drop_height:.6f} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        parser.add_argument(
            "--height",
            type=float,
            default=PARAMS["drop_height"],
            help="Initial sphere bottom height above platform [m].",
        )
        parser.add_argument("--radius", type=float, default=PARAMS["sphere_radius"], help="Sphere radius [m].")
        parser.add_argument("--contact-ke", type=float, default=PARAMS["contact_ke"], help="Contact stiffness [N/m].")
        parser.add_argument(
            "--contact-kd",
            type=float,
            default=PARAMS["contact_kd"],
            help="Contact damping [N*s/m].",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=PARAMS["solver_iterations"],
            help="VBD solver iterations per substep.",
        )
        parser.add_argument(
            "--rigid-contact-hard",
            action="store_true",
            default=PARAMS["rigid_contact_hard"],
            help="Use VBD hard rigid contacts instead of soft penalty contacts.",
        )
        parser.add_argument(
            "--min-rebound-height",
            type=float,
            default=PARAMS["min_rebound_height"],
            help="Minimum peak height recorded as a rebound [m].",
        )
        parser.add_argument("--report", action="store_true", help="Print rebound heights as they are detected.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
