"""Laguerre-Gaussian beam simulation, plotting, and image export.

This module can be imported independently of Tkinter. Run beam_gui.py for
interactive controls, or call generate_and_save_plot from another script:

    from beam import generate_and_save_plot

    generate_and_save_plot(
        "aberrated.png",
        folder="beam_images",
        zernike_terms=[(2, 0, 0.15), (3, 1, 0.08)],
        image_type="aberrated",
    )

image_type="figure" exports all four plots with labels and colorbars.
image_type="ideal" or "aberrated" exports the selected intensity image at the
simulation's grid resolution. All exports use the original colormaps.

Zernike terms are (n, m, coefficient_in_waves). These are unnormalized real
Zernikes: positive m uses cos(m*theta), negative m uses sin(abs(m)*theta), and
the phase contribution is 2*pi*coefficient*Z_n^m. The Zernike radius limits the
phase calculation; it is not a physical aperture that blocks the beam.

Dependencies: numpy, scipy, matplotlib. Tkinter is only needed by beam_gui.py.
"""

import math
import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import eval_genlaguerre
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imsave


# ---------------------------------------------------------------------------
# Optical field calculations: unchanged from the original simulator
# ---------------------------------------------------------------------------

def laguerre_gaussian_field(x, y, w0, p=0, ell=1):
    """Complex LG field at its waist; overall normalization is omitted."""
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
    for k in range((n - m_abs) // 2 + 1):
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


def validate_zernike_indices(n, m):
    """Check that n and m describe an allowed real Zernike polynomial."""
    if n < 0:
        raise ValueError("Zernike radial order n must be >= 0.")
    if abs(m) > n:
        raise ValueError("Zernike indices require |m| <= n.")
    if (n - abs(m)) % 2 != 0:
        raise ValueError("Zernike indices require n - |m| to be even.")


def zernike(n, m, rho, theta):
    """Real, unnormalized Zernike polynomial; zero outside the unit disk."""
    n, m = int(n), int(m)
    validate_zernike_indices(n, m)
    m_abs = abs(m)
    disk = rho <= 1.0
    out = np.zeros_like(rho, dtype=float)
    radial = zernike_radial(n, m_abs, rho)
    if m == 0:
        angular = np.ones_like(theta)
    elif m > 0:
        angular = np.cos(m_abs * theta)
    else:
        angular = np.sin(m_abs * theta)
    out[disk] = (radial * angular)[disk]
    return out


def build_aberration_phase(x, y, aperture_radius, terms):
    """Sum (n, m, coefficient_in_waves) terms into a phase in radians."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    rho = r / aperture_radius
    phase = np.zeros_like(x, dtype=float)
    for n, m, coefficient_waves in terms:
        phase += 2.0 * np.pi * coefficient_waves * zernike(n, m, rho, theta)
    return phase


def angular_spectrum_propagate(field, dx, wavelength, z):
    """Propagate a complex scalar field; dx, wavelength, and z are in meters."""
    if abs(z) < 1e-15:
        return field.copy()
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    k = 2.0 * np.pi / wavelength
    kx, ky = 2.0 * np.pi * FX, 2.0 * np.pi * FY
    kz_squared = k**2 - kx**2 - ky**2
    kz = np.sqrt(kz_squared.astype(complex))
    transfer = np.exp(1j * kz * z)
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * transfer)


def normalized_intensity(field):
    """Return |field|^2 normalized to its own maximum, as in the original."""
    intensity = np.abs(field) ** 2
    max_value = intensity.max()
    if max_value > 0:
        intensity = intensity / max_value
    return intensity


# ---------------------------------------------------------------------------
# Reusable simulation, plot, and export functions
# ---------------------------------------------------------------------------

def _integer(value, name):
    """Accept integer values, including NumPy integers, without truncating."""
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True)
class BeamParameters:
    """Parameters in the same user-facing units as the GUI."""
    wavelength_nm: float = 632.8
    w0_mm: float = 0.80
    p: int = 0
    ell: int = 2
    grid_size: int = 512
    window_mm: float = 8.0
    aperture_mm: float = 2.5
    propagation_m: float = 1.0

    def validate(self):
        for name in ("wavelength_nm", "w0_mm", "window_mm", "aperture_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.propagation_m):
            raise ValueError("propagation_m must be finite.")
        if _integer(self.p, "p") < 0:
            raise ValueError("LG radial index p must be >= 0.")
        _integer(self.ell, "ell")
        if _integer(self.grid_size, "grid_size") < 64:
            raise ValueError("Use at least 64 grid points.")


@dataclass
class SimulationResult:
    """Computed arrays plus the parameters and terms that produced them."""
    parameters: BeamParameters
    zernike_terms: tuple
    x_mm: np.ndarray
    ideal_intensity: np.ndarray
    aberrated_intensity: np.ndarray
    phase_waves: np.ndarray
    intensity_difference: np.ndarray


def simulate_beam(parameters=None, zernike_terms=()):
    """Compute the original simulation without opening windows or saving files.

    Returns SimulationResult. Each intensity is normalized independently to
    its own maximum, preserving the original comparison and difference plot.
    """
    params = parameters if parameters is not None else BeamParameters()
    params.validate()
    terms = []
    for n, m, weight in zernike_terms:
        n, m = _integer(n, "Zernike n"), _integer(m, "Zernike m")
        validate_zernike_indices(n, m)
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("Zernike weights must be finite.")
        terms.append((n, m, weight))
    terms = tuple(terms)

    wavelength = params.wavelength_nm * 1e-9
    w0 = params.w0_mm * 1e-3
    window = params.window_mm * 1e-3
    aperture = params.aperture_mm * 1e-3
    x1d = np.linspace(-window / 2.0, window / 2.0, params.grid_size, endpoint=False)
    dx = x1d[1] - x1d[0]
    X, Y = np.meshgrid(x1d, x1d)
    ideal_input = laguerre_gaussian_field(X, Y, w0, p=params.p, ell=params.ell)
    phase = build_aberration_phase(X, Y, aperture, terms)
    aberrated_input = ideal_input * np.exp(1j * phase)
    ideal_output = angular_spectrum_propagate(
        ideal_input, dx, wavelength, params.propagation_m
    )
    aberrated_output = angular_spectrum_propagate(
        aberrated_input, dx, wavelength, params.propagation_m
    )
    ideal = normalized_intensity(ideal_output)
    aberrated = normalized_intensity(aberrated_output)
    return SimulationResult(
        parameters=params,
        zernike_terms=terms,
        x_mm=x1d * 1e3,
        ideal_intensity=ideal,
        aberrated_intensity=aberrated,
        phase_waves=phase / (2.0 * np.pi),
        intensity_difference=aberrated - ideal,
    )


def create_plot(result, figure=None):
    """Draw the original four-panel plot into a new or existing Figure.

    This does not show a window or save a file. When reusing a GUI figure,
    clearing the entire figure also removes its previous colorbar axes.
    """
    if figure is None:
        figure = Figure(figsize=(10, 8), constrained_layout=True)
        FigureCanvasAgg(figure)
    else:
        figure.clear()
    axes = figure.subplots(2, 2)
    params = result.parameters
    extent = [result.x_mm[0], result.x_mm[-1], result.x_mm[0], result.x_mm[-1]]
    phase_limit = np.max(np.abs(result.phase_waves)) or 1.0
    diff_limit = np.max(np.abs(result.intensity_difference)) or 1.0
    panels = (
        (result.ideal_intensity,
         f"Ideal LG intensity at z = {params.propagation_m:g} m", 0, 1, "gray"),
        (result.phase_waves, "Applied Zernike phase [waves]",
         -phase_limit, phase_limit, "RdBu_r"),
        (result.aberrated_intensity, "Aberrated LG intensity", 0, 1, "gray"),
        (result.intensity_difference, "Aberrated - ideal normalized intensity",
         -diff_limit, diff_limit, "RdBu_r"),
    )
    for ax, (data, title, vmin, vmax, cmap) in zip(axes.flat, panels):
        image = ax.imshow(
            data, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"LG(p={params.p}, ell={params.ell}), w0={params.w0_mm:.3g} mm, "
        f"lambda={params.wavelength_nm:.4g} nm"
    )
    return figure


def save_simulation_image(result, output_path, *, image_type="figure", dpi=300,
                          figure=None, folder=None):
    """Save a completed simulation and return the absolute output Path.

    image_type is 'figure', 'ideal', or 'aberrated'. Beam-only images have
    grid_size x grid_size pixels; dpi changes the full figure's resolution.
    Missing folders are created. An existing destination is overwritten.
    The optional figure lets the GUI save the already displayed full plot.
    Optional folder accepts a string or Path and overrides the directory
    in output_path; its filename is preserved for every image type.
    """
    if image_type not in ("figure", "ideal", "aberrated"):
        raise ValueError("image_type must be 'figure', 'ideal', or 'aberrated'.")
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive.")
    path = Path(output_path).expanduser()
    if folder is not None:
        path = Path(folder).expanduser() / path.name
    if not path.suffix:
        path = path.with_suffix(".png")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_type == "figure":
        owns_figure = figure is None
        if owns_figure:
            figure = create_plot(result)
        try:
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            if owns_figure:
                figure.clear()
    else:
        data = (result.ideal_intensity if image_type == "ideal"
                else result.aberrated_intensity)
        image_format = "tiff" if path.suffix.lower() in (".tif", ".tiff") else None
        """imsave(
            path, data, cmap="gray", origin="lower", vmin=0, vmax=1,
            format=image_format,
        )"""
    return path, image_format, data


def generate_and_save_plot(
    output_path,
    *,
    wavelength_nm=632.8,
    w0_mm=0.80,
    p=0,
    ell=2,
    grid_size=512,
    window_mm=8.0,
    aperture_mm=2.5,
    propagation_m=1.0,
    zernike_terms=(),
    image_type="figure",
    dpi=300,
    folder : Path,
):
    """Generate a beam simulation from parameters and save it as an image.

    No GUI opens. output_path can be a string or pathlib.Path. Numeric inputs
    use nm for wavelength, mm for waist/window/Zernike radius, meters for z,
    and waves for Zernike weights. zernike_terms is any iterable of (n, m,
    coefficient_in_waves); an empty iterable means no aberration.

    Optional folder overrides the directory in output_path. Alternatively,
    include the desired directory directly in output_path.

    Return value: absolute pathlib.Path of the saved image.
    """
    parameters = BeamParameters(
        wavelength_nm=wavelength_nm, w0_mm=w0_mm, p=p, ell=ell,
        grid_size=grid_size, window_mm=window_mm, aperture_mm=aperture_mm,
        propagation_m=propagation_m,
    )
    result = simulate_beam(parameters, zernike_terms)
    return save_simulation_image(
        result, output_path, image_type=image_type, dpi=dpi, folder=folder
    )
