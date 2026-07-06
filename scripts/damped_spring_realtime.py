#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Realtime analytic solver for a damped spring under constant force.

Run directly:

    ./scripts/damped_spring_realtime.py
"""

from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk


@dataclass(frozen=True)
class Parameters:
    mass: float
    damping: float
    stiffness: float
    force: float
    x0: float
    v0: float
    t_max: float
    time_step: float


@dataclass(frozen=True)
class SolutionInfo:
    omega_n: float
    zeta: float
    regime: str
    x_eq: float
    sample_count: int
    actual_time_step: float


class DampedSpringApp(tk.Tk):
    """Tkinter UI for plotting the closed-form damped spring response."""

    _WIDTH = 1360
    _HEIGHT = 680
    _PAD_LEFT = 92
    _PAD_RIGHT = 44
    _PAD_TOP = 42
    _PAD_BOTTOM = 74
    _MAX_SAMPLES = 50_000

    def __init__(self) -> None:
        super().__init__()
        self.title("Damped Spring Realtime Solver")
        self.geometry("1400x920")
        self.minsize(1180, 820)

        self._controls: dict[str, tk.DoubleVar] = {}
        self._pending_update: str | None = None

        self._configure_display()
        self._build_ui()
        self._schedule_update()

    def _configure_display(self) -> None:
        self.tk.call("tk", "scaling", 1.35)
        font_sizes = {
            "TkDefaultFont": 12,
            "TkTextFont": 12,
            "TkMenuFont": 12,
            "TkHeadingFont": 13,
            "TkFixedFont": 12,
        }
        for name, size in font_sizes.items():
            tkfont.nametofont(name).configure(size=size)

        style = ttk.Style(self)
        style.configure(".", font=tkfont.nametofont("TkDefaultFont"))
        style.configure("TButton", padding=(14, 8))
        style.configure("TSpinbox", padding=(6, 4))

        self._axis_font = tkfont.Font(family="TkDefaultFont", size=12)
        self._label_font = tkfont.Font(family="TkDefaultFont", size=13)
        self._info_font = tkfont.Font(family="TkDefaultFont", size=13)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, width=self._WIDTH, height=self._HEIGHT, bg="#ffffff", highlightthickness=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._schedule_update())
        root.rowconfigure(0, weight=1)

        self.info_var = tk.StringVar()
        info = ttk.Label(root, textvariable=self.info_var, anchor="w", font=self._info_font)
        info.grid(row=1, column=0, sticky="ew", pady=(12, 14))

        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, sticky="ew")
        for col in range(3):
            controls.columnconfigure(col, weight=1)

        specs = [
            ("mass", "m [kg]", 0.1, 20.0, 1.0, 0.1),
            ("damping", "kd [N*s/m]", 0.0, 1000.0, 200.0, 1.0),
            ("stiffness", "k [N/m]", 0.1, 200000.0, 100000.0, 100.0),
            ("force", "F [N]", -80.0, 80.0, 10.0, 0.5),
            ("x0", "x(0) [m]", -10.0, 10.0, 0.0, 0.05),
            ("v0", "v(0) [m/s]", -30.0, 30.0, 4.0, 0.1),
            ("t_max", "time window [s]", 0.01, 30.0, 1.0, 0.01),
            ("time_step", "dt [s]", 0.0001, 0.2, 0.001, 0.0001),
        ]
        for index, spec in enumerate(specs):
            self._add_control(controls, index, *spec)

        ttk.Button(controls, text="Reset", command=self._reset).grid(row=3, column=2, sticky="e", padx=8, pady=(16, 0))

    def _add_control(
        self,
        parent: ttk.Frame,
        index: int,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
    ) -> None:
        row = index // 3
        col = index % 3
        frame = ttk.Frame(parent, padding=(8, 4))
        frame.grid(row=row, column=col, sticky="ew")
        frame.columnconfigure(1, weight=1)

        var = tk.DoubleVar(value=value)
        self._controls[key] = var

        ttk.Label(frame, text=label, width=16).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(frame, from_=minimum, to=maximum, variable=var, command=lambda _value: self._schedule_update())
        scale.grid(row=0, column=1, sticky="ew", padx=(12, 12))

        spin = ttk.Spinbox(
            frame,
            from_=minimum,
            to=maximum,
            increment=step,
            textvariable=var,
            width=10,
            command=self._schedule_update,
        )
        spin.grid(row=0, column=2, sticky="e")
        spin.bind("<KeyRelease>", lambda _event: self._schedule_update())
        spin.bind("<FocusOut>", lambda _event: self._schedule_update())

    def _reset(self) -> None:
        defaults = {
            "mass": 1.0,
            "damping": 200.0,
            "stiffness": 100000.0,
            "force": 10.0,
            "x0": 0.0,
            "v0": 4.0,
            "t_max": 1.0,
            "time_step": 0.001,
        }
        for key, value in defaults.items():
            self._controls[key].set(value)
        self._schedule_update()

    def _schedule_update(self) -> None:
        if self._pending_update is not None:
            self.after_cancel(self._pending_update)
        self._pending_update = self.after(20, self._update_plot)

    def _read_parameters(self) -> Parameters:
        values = {key: var.get() for key, var in self._controls.items()}
        return Parameters(
            mass=max(values["mass"], 1.0e-9),
            damping=max(values["damping"], 0.0),
            stiffness=max(values["stiffness"], 1.0e-9),
            force=values["force"],
            x0=values["x0"],
            v0=values["v0"],
            t_max=max(values["t_max"], 1.0e-6),
            time_step=max(values["time_step"], 1.0e-6),
        )

    def _solve(self, params: Parameters) -> tuple[list[tuple[float, float]], SolutionInfo]:
        omega_n = math.sqrt(params.stiffness / params.mass)
        zeta = params.damping / (2.0 * math.sqrt(params.mass * params.stiffness))
        x_eq = params.force / params.stiffness
        y0 = params.x0 - x_eq

        if abs(zeta - 1.0) < 1.0e-4:
            c1 = y0
            c2 = params.v0 + omega_n * c1
            regime = "critical damping"

            def x_at(t: float) -> float:
                return x_eq + math.exp(-omega_n * t) * (c1 + c2 * t)

        elif zeta < 1.0:
            omega_d = omega_n * math.sqrt(max(0.0, 1.0 - zeta * zeta))
            c1 = y0
            c2 = (params.v0 + zeta * omega_n * c1) / omega_d
            regime = "underdamped"

            def x_at(t: float) -> float:
                envelope = math.exp(-zeta * omega_n * t)
                return x_eq + envelope * (c1 * math.cos(omega_d * t) + c2 * math.sin(omega_d * t))

        else:
            root = omega_n * math.sqrt(zeta * zeta - 1.0)
            lambda_1 = -zeta * omega_n + root
            lambda_2 = -zeta * omega_n - root
            c1 = (params.v0 - lambda_2 * y0) / (lambda_1 - lambda_2)
            c2 = y0 - c1
            regime = "overdamped"

            def x_at(t: float) -> float:
                return x_eq + c1 * math.exp(lambda_1 * t) + c2 * math.exp(lambda_2 * t)

        sample_count = min(max(2, math.ceil(params.t_max / params.time_step)), self._MAX_SAMPLES)
        actual_time_step = params.t_max / sample_count

        points = []
        for i in range(sample_count + 1):
            t = actual_time_step * i
            points.append((t, x_at(t)))
        return points, SolutionInfo(
            omega_n=omega_n,
            zeta=zeta,
            regime=regime,
            x_eq=x_eq,
            sample_count=sample_count,
            actual_time_step=actual_time_step,
        )

    def _update_plot(self) -> None:
        self._pending_update = None
        params = self._read_parameters()
        points, info = self._solve(params)

        self.info_var.set(
            f"omega_n={info.omega_n:.4g} rad/s    zeta={info.zeta:.4g}    "
            f"regime={info.regime}    equilibrium F/k={info.x_eq:.4g} m    "
            f"dt={info.actual_time_step:.4g} s    samples={info.sample_count + 1}"
        )

        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), self._WIDTH)
        height = max(self.canvas.winfo_height(), self._HEIGHT)
        left = self._PAD_LEFT
        right = width - self._PAD_RIGHT
        top = self._PAD_TOP
        bottom = height - self._PAD_BOTTOM

        x_values = [x for _t, x in points]
        y_min = min(min(x_values), params.x0, info.x_eq)
        y_max = max(max(x_values), params.x0, info.x_eq)
        if math.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding

        def sx(t: float) -> float:
            return left + (right - left) * (t / params.t_max)

        def sy(x: float) -> float:
            return bottom - (bottom - top) * ((x - y_min) / (y_max - y_min))

        self._draw_grid(left, right, top, bottom, params.t_max, y_min, y_max, sx, sy)

        if y_min <= info.x_eq <= y_max:
            y_eq = sy(info.x_eq)
            self.canvas.create_line(left, y_eq, right, y_eq, fill="#475569", dash=(7, 5), width=2)
            self.canvas.create_text(
                right - 8,
                y_eq - 16,
                text="F/k",
                anchor="e",
                fill="#334155",
                font=self._axis_font,
            )

        flat_points: list[float] = []
        for t, x in points:
            flat_points.extend((sx(t), sy(x)))
        self.canvas.create_line(*flat_points, fill="#0f766e", width=4, smooth=True)

        self.canvas.create_oval(sx(0.0) - 6, sy(params.x0) - 6, sx(0.0) + 6, sy(params.x0) + 6, fill="#dc2626", outline="")
        self.canvas.create_text(left, bottom + 42, text="t [s]", anchor="w", fill="#1e293b", font=self._label_font)
        self.canvas.create_text(18, top, text="x [m]", anchor="nw", fill="#1e293b", font=self._label_font)

    def _draw_grid(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        t_max: float,
        y_min: float,
        y_max: float,
        sx: Callable[[float], float],
        sy: Callable[[float], float],
    ) -> None:
        self.canvas.create_rectangle(left, top, right, bottom, outline="#64748b", width=2)

        for i in range(6):
            t = t_max * i / 5
            x = sx(t)
            self.canvas.create_line(x, top, x, bottom, fill="#e2e8f0")
            self.canvas.create_text(x, bottom + 22, text=f"{t:.2g}", anchor="n", fill="#334155", font=self._axis_font)

        for i in range(6):
            value = y_min + (y_max - y_min) * i / 5
            y = sy(value)
            self.canvas.create_line(left, y, right, y, fill="#e2e8f0")
            self.canvas.create_text(left - 12, y, text=f"{value:.3g}", anchor="e", fill="#334155", font=self._axis_font)


def main() -> int:
    app = DampedSpringApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
