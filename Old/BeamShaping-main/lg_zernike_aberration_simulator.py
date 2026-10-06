"""
 Laguerre-Gaussian beam with Zernike aberration simulation.



Zernike convention used here:
    - n >= 0
    - |m| <= n
    - n - |m| must be even
    - m >= 0 -> cos(m*theta)
    - m < 0  -> sin(|m|*theta)

Each Zernike coefficient is entered in waves of phase:
    phase contribution = 2*pi*coefficient*Z_n^m
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------------------------------------------------------------
# Optical field calculations
# ---------------------------------------------------------------------------

def laguerre_gaussian_field(x, y, w0, p=0, ell=1):
    """
    Complex Laguerre-Gaussian field at its waist plane.

    The overall normalization constant is omitted because all displayed
    intensities are normalized.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    abs_ell = abs(int(ell))

    rho2 = 2.0 * r**2 / w0**2
    radial = (
        (np.sqrt(2.0) * r / w0) ** abs_ell
        * eval_genlaguerre(int(p), abs_ell, rho2)
        * np.exp(-r**2 / w0**2)
    )
    return radial * np.exp(1j * int(ell) * theta)


def zernike_radial(n, m_abs, rho):
    """Unnormalized Zernike radial polynomial R_n^m(rho)."""
    if n < 0 or m_abs < 0 or m_abs > n or (n - m_abs) % 2 != 0:
        raise ValueError("Invalid Zernike indices.")

    result = np.zeros_like(rho, dtype=float)
    upper = (n - m_abs) // 2

    for k in range(upper + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        result += coeff * rho ** (n - 2 * k)

    return result


def zernike(n, m, rho, theta):
    """
    Real-valued, unnormalized Zernike polynomial on the unit disk.

    Convention:
      m >= 0 : R_n^|m| cos(|m| theta)
      m <  0 : R_n^|m| sin(|m| theta)

    The returned value is zero outside rho > 1.
    """
    n = int(n)
    m = int(m)
    m_abs = abs(m)

    if n < 0:
        raise ValueError("n must be non-negative.")
    if m_abs > n:
        raise ValueError("|m| must be <= n.")
    if (n - m_abs) % 2 != 0:
        raise ValueError("n - |m| must be even.")

    disk = rho <= 1.0
    out = np.zeros_like(rho, dtype=float)
    R = zernike_radial(n, m_abs, rho)

    if m == 0:
        angular = np.ones_like(theta)
    elif m > 0:
        angular = np.cos(m_abs * theta)
    else:
        angular = np.sin(m_abs * theta)

    out[disk] = (R * angular)[disk]
    return out


def build_aberration_phase(x, y, aperture_radius, terms):
    """
    Build total Zernike phase in radians.

    terms is an iterable of (n, m, coefficient_in_waves).
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    rho = r / aperture_radius

    phase = np.zeros_like(x, dtype=float)
    for n, m, coefficient_waves in terms:
        phase += 2.0 * np.pi * coefficient_waves * zernike(
            n, m, rho, theta
        )

    return phase


def angular_spectrum_propagate(field, dx, wavelength, z):
    """
    Propagate a scalar complex field using the angular-spectrum method.

    field      : sampled complex field
    dx         : sample pitch [m], assumed equal in x and y
    wavelength : wavelength [m]
    z          : propagation distance [m]
    """
    if abs(z) < 1e-15:
        return field.copy()

    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY

    kz_squared = k**2 - kx**2 - ky**2

    # Complex square root automatically handles evanescent components.
    kz = np.sqrt(kz_squared.astype(complex))
    transfer = np.exp(1j * kz * z)

    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * transfer)


def normalized_intensity(field):
    """Return |field|^2 normalized to its own maximum."""
    intensity = np.abs(field) ** 2
    max_value = intensity.max()
    if max_value > 0:
        intensity = intensity / max_value
    return intensity


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class LGZernikeApp(tk.Tk):
    PRESETS = {
        "Piston": (0, 0),
        "Tilt X": (1, 1),
        "Tilt Y": (1, -1),
        "Defocus": (2, 0),
        "Astigmatism X": (2, 2),
        "Astigmatism Y": (2, -2),
        "Coma X": (3, 1),
        "Coma Y": (3, -1),
        "Trefoil X": (3, 3),
        "Trefoil Y": (3, -3),
        "Primary spherical": (4, 0),
    }

    def __init__(self):
        super().__init__()
        self.title("Laguerre-Gaussian Beam with Zernike Aberrations")
        self.geometry("1500x900")
        self.minsize(1200, 760)

        self._build_controls()
        self._build_plot_area()

        # Start with a small example aberration so the initial plot visibly
        # demonstrates the program.
        self._insert_term(2, 0, 0.15)
        self._insert_term(3, 1, 0.08)

        self.after(150, self.update_plots)

    def _build_controls(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        # Beam parameters
        beam_frame = ttk.LabelFrame(controls, text="Laguerre-Gaussian beam", padding=8)
        beam_frame.pack(fill=tk.X, pady=(0, 8))

        self.wavelength_nm = tk.StringVar(value="632.8")
        self.w0_mm = tk.StringVar(value="0.80")
        self.p_index = tk.StringVar(value="0")
        self.ell_index = tk.StringVar(value="2")

        self._labeled_entry(beam_frame, "Wavelength [nm]", self.wavelength_nm, 0)
        self._labeled_entry(beam_frame, "Waist w0 [mm]", self.w0_mm, 1)
        self._labeled_entry(beam_frame, "Radial index p", self.p_index, 2)
        self._labeled_entry(beam_frame, "Azimuthal index ell", self.ell_index, 3)

        # Simulation parameters
        sim_frame = ttk.LabelFrame(controls, text="Simulation", padding=8)
        sim_frame.pack(fill=tk.X, pady=(0, 8))

        self.grid_size = tk.StringVar(value="512")
        self.window_mm = tk.StringVar(value="8.0")
        self.aperture_mm = tk.StringVar(value="2.5")
        self.propagation_m = tk.StringVar(value="1.0")

        self._labeled_entry(sim_frame, "Grid points N", self.grid_size, 0)
        self._labeled_entry(sim_frame, "Window width [mm]", self.window_mm, 1)
        self._labeled_entry(sim_frame, "Zernike radius [mm]", self.aperture_mm, 2)
        self._labeled_entry(sim_frame, "Propagation z [m]", self.propagation_m, 3)

        # Zernike entry
        z_frame = ttk.LabelFrame(controls, text="Add Zernike aberration", padding=8)
        z_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(z_frame, text="Common aberration").grid(
            row=0, column=0, sticky="w", padx=3, pady=3
        )
        self.preset_var = tk.StringVar(value="Defocus")
        preset_box = ttk.Combobox(
            z_frame,
            textvariable=self.preset_var,
            values=list(self.PRESETS.keys()),
            state="readonly",
            width=20,
        )
        preset_box.grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        preset_box.bind("<<ComboboxSelected>>", self._preset_selected)

        self.n_var = tk.StringVar(value="2")
        self.m_var = tk.StringVar(value="0")
        self.weight_var = tk.StringVar(value="0.10")

        self._labeled_entry(z_frame, "Radial order n", self.n_var, 1)
        self._labeled_entry(z_frame, "Azimuthal order m", self.m_var, 2)
        self._labeled_entry(z_frame, "Weight [waves]", self.weight_var, 3)

        ttk.Button(z_frame, text="Add term", command=self.add_term).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=3, pady=(6, 3)
        )

        # Zernike term list
        list_frame = ttk.LabelFrame(controls, text="Active Zernike terms", padding=8)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.term_tree = ttk.Treeview(
            list_frame,
            columns=("n", "m", "weight", "name"),
            show="headings",
            height=10,
        )
        self.term_tree.heading("n", text="n")
        self.term_tree.heading("m", text="m")
        self.term_tree.heading("weight", text="Weight [waves]")
        self.term_tree.heading("name", text="Type")

        self.term_tree.column("n", width=35, anchor="center")
        self.term_tree.column("m", width=35, anchor="center")
        self.term_tree.column("weight", width=95, anchor="center")
        self.term_tree.column("name", width=130, anchor="w")
        self.term_tree.pack(fill=tk.BOTH, expand=True)

        button_row = ttk.Frame(list_frame)
        button_row.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(
            button_row, text="Remove selected", command=self.remove_selected
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        ttk.Button(
            button_row, text="Clear all", command=self.clear_terms
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Main action
        ttk.Button(
            controls,
            text="Update plots",
            command=self.update_plots,
        ).pack(fill=tk.X, ipady=5)

        ttk.Label(
            controls,
            text=(
                "Zernike coefficients are phase weights in waves.\n"
                "Positive m uses cos(mθ); negative m uses sin(|m|θ).\n"
                "Zernikes are evaluated inside the specified radius."
            ),
            justify=tk.LEFT,
            wraplength=320,
        ).pack(fill=tk.X, pady=(8, 0))

    def _build_plot_area(self):
        plot_frame = ttk.Frame(self, padding=(0, 10, 10, 10))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

        # Keep explicit references to colorbars so they can be removed before
        # each redraw. Clearing an image axis does NOT automatically remove
        # the separate axes Matplotlib creates for its colorbar.
        self.colorbars = []

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _labeled_entry(parent, label, variable, row):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=3, pady=3
        )
        entry = ttk.Entry(parent, textvariable=variable, width=14)
        entry.grid(row=row, column=1, sticky="ew", padx=3, pady=3)
        parent.columnconfigure(1, weight=1)
        return entry

    def _preset_selected(self, event=None):
        n, m = self.PRESETS[self.preset_var.get()]
        self.n_var.set(str(n))
        self.m_var.set(str(m))

    def _name_for_indices(self, n, m):
        for name, pair in self.PRESETS.items():
            if pair == (n, m):
                return name
        return f"Z({n},{m})"

    def _insert_term(self, n, m, weight):
        self.term_tree.insert(
            "",
            tk.END,
            values=(n, m, f"{weight:.6g}", self._name_for_indices(n, m)),
        )

    def add_term(self):
        try:
            n = int(self.n_var.get())
            m = int(self.m_var.get())
            weight = float(self.weight_var.get())
            self._validate_zernike_indices(n, m)
        except ValueError as exc:
            messagebox.showerror("Invalid Zernike term", str(exc))
            return

        self._insert_term(n, m, weight)
        self.update_plots()

    def remove_selected(self):
        for item in self.term_tree.selection():
            self.term_tree.delete(item)
        self.update_plots()

    def clear_terms(self):
        for item in self.term_tree.get_children():
            self.term_tree.delete(item)
        self.update_plots()

    @staticmethod
    def _validate_zernike_indices(n, m):
        if n < 0:
            raise ValueError("Zernike radial order n must be >= 0.")
        if abs(m) > n:
            raise ValueError("Zernike indices require |m| <= n.")
        if (n - abs(m)) % 2 != 0:
            raise ValueError("Zernike indices require n - |m| to be even.")

    def _read_terms(self):
        terms = []
        for item in self.term_tree.get_children():
            values = self.term_tree.item(item, "values")
            n = int(values[0])
            m = int(values[1])
            weight = float(values[2])
            terms.append((n, m, weight))
        return terms

    def _read_parameters(self):
        wavelength = float(self.wavelength_nm.get()) * 1e-9
        w0 = float(self.w0_mm.get()) * 1e-3
        p = int(self.p_index.get())
        ell = int(self.ell_index.get())
        N = int(self.grid_size.get())
        window = float(self.window_mm.get()) * 1e-3
        aperture = float(self.aperture_mm.get()) * 1e-3
        z = float(self.propagation_m.get())

        if wavelength <= 0:
            raise ValueError("Wavelength must be positive.")
        if w0 <= 0:
            raise ValueError("Beam waist must be positive.")
        if p < 0:
            raise ValueError("LG radial index p must be >= 0.")
        if N < 64:
            raise ValueError("Use at least 64 grid points.")
        if window <= 0:
            raise ValueError("Simulation window must be positive.")
        if aperture <= 0:
            raise ValueError("Zernike radius must be positive.")

        return wavelength, w0, p, ell, N, window, aperture, z

    def update_plots(self):
        try:
            wavelength, w0, p, ell, N, window, aperture, z = (
                self._read_parameters()
            )
            terms = self._read_terms()

            x1d = np.linspace(-window / 2.0, window / 2.0, N, endpoint=False)
            dx = x1d[1] - x1d[0]
            X, Y = np.meshgrid(x1d, x1d)

            ideal_input = laguerre_gaussian_field(X, Y, w0, p=p, ell=ell)
            aberration_phase = build_aberration_phase(
                X, Y, aperture, terms
            )
            aberrated_input = ideal_input * np.exp(1j * aberration_phase)

            ideal_output = angular_spectrum_propagate(
                ideal_input, dx, wavelength, z
            )
            aberrated_output = angular_spectrum_propagate(
                aberrated_input, dx, wavelength, z
            )

            I_ideal = normalized_intensity(ideal_output)
            I_ab = normalized_intensity(aberrated_output)
            difference = I_ab - I_ideal

            extent = [
                x1d[0] * 1e3,
                x1d[-1] * 1e3,
                x1d[0] * 1e3,
                x1d[-1] * 1e3,
            ]

            # Remove colorbars from the previous update before clearing the
            # image axes. Matplotlib colorbars live in their own Axes objects,
            # so ax.clear() alone would leave the old colorbar axes behind.
            for cbar in self.colorbars:
                cbar.remove()
            self.colorbars.clear()

            for ax in self.axes.flat:
                ax.clear()

            im0 = self.axes[0, 0].imshow(
                I_ideal,
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
            self.axes[0, 0].set_title(f"Ideal LG intensity at z = {z:g} m")
            self.axes[0, 0].set_xlabel("x [mm]")
            self.axes[0, 0].set_ylabel("y [mm]")
            self.colorbars.append(
                self.fig.colorbar(
                    im0, ax=self.axes[0, 0], fraction=0.046, pad=0.04
                )
            )

            phase_waves = aberration_phase / (2.0 * np.pi)
            max_phase = np.max(np.abs(phase_waves))
            if max_phase == 0:
                max_phase = 1.0

            im1 = self.axes[0, 1].imshow(
                phase_waves,
                origin="lower",
                extent=extent,
                vmin=-max_phase,
                vmax=max_phase,
                cmap="RdBu_r",
            )
            self.axes[0, 1].set_title("Applied Zernike phase [waves]")
            self.axes[0, 1].set_xlabel("x [mm]")
            self.axes[0, 1].set_ylabel("y [mm]")
            self.colorbars.append(
                self.fig.colorbar(
                    im1, ax=self.axes[0, 1], fraction=0.046, pad=0.04
                )
            )

            im2 = self.axes[1, 0].imshow(
                I_ab,
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
            self.axes[1, 0].set_title("Aberrated LG intensity")
            self.axes[1, 0].set_xlabel("x [mm]")
            self.axes[1, 0].set_ylabel("y [mm]")
            self.colorbars.append(
                self.fig.colorbar(
                    im2, ax=self.axes[1, 0], fraction=0.046, pad=0.04
                )
            )

            diff_limit = np.max(np.abs(difference))
            if diff_limit == 0:
                diff_limit = 1.0

            im3 = self.axes[1, 1].imshow(
                difference,
                origin="lower",
                extent=extent,
                vmin=-diff_limit,
                vmax=diff_limit,
                cmap="RdBu_r",
            )
            self.axes[1, 1].set_title("Aberrated - ideal normalized intensity")
            self.axes[1, 1].set_xlabel("x [mm]")
            self.axes[1, 1].set_ylabel("y [mm]")
            self.colorbars.append(
                self.fig.colorbar(
                    im3, ax=self.axes[1, 1], fraction=0.046, pad=0.04
                )
            )

            self.fig.suptitle(
                f"LG(p={p}, ell={ell}), w0={w0*1e3:.3g} mm, "
                f"lambda={wavelength*1e9:.4g} nm"
            )

            self.canvas.draw_idle()

        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))


if __name__ == "__main__":
    app = LGZernikeApp()
    app.mainloop()
