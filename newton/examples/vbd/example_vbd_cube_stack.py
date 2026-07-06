# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Cube Stack
#
# Builds a small rigid-body cube stack and solves it with SolverVBD.
#
# Command: python -m newton.examples vbd_cube_stack
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


PARAMS = {
    "fps": 60,
    "sim_substeps": 5,
    "solver_iterations": 10,
    "rigid_contact_hard": True,
    "rigid_contact_history": False,
    "rigid_avbd_alpha": 0.0,
    "gravity": -9.81,
    "cube_half_extent": 0.01,
    "cube_gap": 0.001,
    "level_gap": 0.006,
    "stack_drop_height": 1.0,
    "stack_levels": 1,
    "platform_half_extent": 0.6,
    "platform_thickness": 0.04,
    "cube_density": 1000.0,
    "cube_ke": 300.0,
    "cube_kd": 0.0,
    "cube_mu": 0.85,
    "platform_ke": 300,
    "platform_kd": 0.0,
    "platform_mu": 0.9,
    "shape_margin": 0.000,
    "rigid_contact_max": 8192,
    "rigid_body_contact_buffer_size": 4096,
}


def build_model(builder, params):
    cube_half = params["cube_half_extent"]
    cube_size = cube_half * 2.0
    xy_spacing = cube_size + params["cube_gap"]
    z_spacing = cube_size + params["level_gap"]
    platform_top = 0.0

    platform_cfg = newton.ModelBuilder.ShapeConfig()
    platform_cfg.ke = params["platform_ke"]
    platform_cfg.kd = params["platform_kd"]
    platform_cfg.mu = params["platform_mu"]
    platform_cfg.margin = params["shape_margin"]

    platform_half = params["platform_half_extent"]
    platform_thickness = params["platform_thickness"]
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, platform_top - platform_thickness * 0.5), wp.quat_identity()),
        hx=platform_half,
        hy=platform_half,
        hz=platform_thickness * 0.5,
        cfg=platform_cfg,
        color=wp.vec3(0.45, 0.48, 0.50),
        label="platform",
    )

    cube_cfg = newton.ModelBuilder.ShapeConfig()
    cube_cfg.density = params["cube_density"]
    cube_cfg.ke = params["cube_ke"]
    cube_cfg.kd = params["cube_kd"]
    cube_cfg.mu = params["cube_mu"]
    cube_cfg.margin = params["shape_margin"]

    cube_bodies = []
    levels = params["stack_levels"]
    palette = [
        wp.vec3(0.82, 0.30, 0.25),
        wp.vec3(0.22, 0.50, 0.78),
        wp.vec3(0.28, 0.62, 0.42),
        wp.vec3(0.92, 0.68, 0.22),
    ]

    for level in range(levels):
        cubes_per_side = levels - level
        row_width = (cubes_per_side - 1) * xy_spacing
        z_pos = platform_top + params["stack_drop_height"] + cube_half + level * z_spacing

        for x_idx in range(cubes_per_side):
            for y_idx in range(cubes_per_side):
                x_pos = -row_width * 0.5 + x_idx * xy_spacing
                y_pos = -row_width * 0.5 + y_idx * xy_spacing
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(x_pos, y_pos, z_pos), wp.quat_identity()),
                    label=f"cube_{level}_{x_idx}_{y_idx}",
                )
                builder.add_shape_box(
                    body,
                    hx=cube_half,
                    hy=cube_half,
                    hz=cube_half,
                    cfg=cube_cfg,
                    color=palette[level % len(palette)],
                )
                cube_bodies.append(body)

    builder.color()

    return {
        "cube_bodies": cube_bodies,
        "platform_top": platform_top,
        "platform_half_extent": platform_half,
    }


def setup_sim(builder, params):
    model = builder.finalize()
    model.rigid_contact_max = params["rigid_contact_max"]
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=params["solver_iterations"],
        rigid_body_contact_buffer_size=params["rigid_body_contact_buffer_size"],
        rigid_avbd_alpha=params["rigid_avbd_alpha"],
        rigid_contact_hard=params["rigid_contact_hard"],
        rigid_contact_history=params["rigid_contact_history"],
    )
    return model, solver


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS.copy()
        self.params["stack_levels"] = args.stack_levels
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = build_model(builder, self.params)
        self.model, self.solver = setup_sim(builder, self.params)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.65, -0.75, 0.55), -18.0, 138.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        cube_indices = self.info["cube_bodies"]
        cube_half = self.params["cube_half_extent"]
        min_z = self.info["platform_top"] + cube_half * 0.25
        max_xy = self.info["platform_half_extent"] + cube_half

        cube_q = body_q[cube_indices]
        if not np.all(np.isfinite(cube_q)):
            raise ValueError("Cube stack contains non-finite body transforms")

        low = cube_q[:, 2] < min_z
        if np.any(low):
            failed = [cube_indices[i] for i in np.where(low)[0]]
            raise ValueError(f"Cube bodies fell through the platform: {failed}")

        escaped = (np.abs(cube_q[:, 0]) > max_xy) | (np.abs(cube_q[:, 1]) > max_xy)
        if np.any(escaped):
            failed = [cube_indices[i] for i in np.where(escaped)[0]]
            raise ValueError(f"Cube bodies moved outside the platform bounds: {failed}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument(
            "--stack-levels",
            type=int,
            default=PARAMS["stack_levels"],
            help="Number of square layers in the initial cube stack.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
