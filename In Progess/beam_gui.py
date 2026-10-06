"""Tkinter interface for the Laguerre-Gaussian and Zernike beam simulator.

Keep this file in the same directory as beam.py and run:

    python beam_gui.py

All beam calculations, plotting, and image export are provided by beam.py.
This file handles the controls, embedded plot canvas, dialogs, and window.

Dependencies: numpy, scipy, matplotlib, and tkinter.
"""

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beam import (
    BeamParameters,
    create_plot,
    save_simulation_image,
    simulate_beam,
    validate_zernike_indices,
)


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

        # Leave room for the taskbar and window borders on smaller displays.
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(1400, max(320, screen_width - 80))
        height = min(850, max(320, screen_height - 120))
        self.geometry(f"{width}x{height}+20+20")
        self.minsize(min(900, width), min(480, height))

        self.simulation_result = None

        self._build_controls()

        self._build_plot_area()

        # Start with a small example aberration so the initial plot visibly

        # demonstrates the program.

        self._insert_term(2, 0, 0.15)

        self._insert_term(3, 1, 0.08)

        self.after(150, self.update_plots)

    def _build_controls(self):
        # Measure the current UI font so a narrow sidebar still fits scaled text.
        font = tkfont.nametofont("TkDefaultFont")
        label_width = max(font.measure("Common aberration"),
                          font.measure("Azimuthal index ell"))
        field_width = font.measure("Primary spherical") + 30
        sidebar_width = max(280, label_width + field_width + 45)
        sidebar = ttk.Frame(self, width=sidebar_width, padding=5)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # These two main actions remain visible while the other controls scroll.
        actions = ttk.Frame(sidebar, padding=(0, 4, 0, 0))
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(actions, text="Update plots", command=self.update_plots).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        self.save_button = ttk.Button(
            actions, text="Save image", command=self.save_plot_image,
            state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        viewport = ttk.Frame(sidebar)
        viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        background = ttk.Style(self).lookup("TFrame", "background") or self.cget("bg")
        self.controls_canvas = tk.Canvas(
            viewport, width=sidebar_width - 20, highlightthickness=0,
            background=background, yscrollincrement=18,
        )
        scrollbar = ttk.Scrollbar(
            viewport, orient=tk.VERTICAL, command=self.controls_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        controls = ttk.Frame(self.controls_canvas, padding=(1, 1, 3, 3))
        self.controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls)

        beam_frame = ttk.LabelFrame(controls, text="Laguerre-Gaussian beam", padding=4)
        beam_frame.pack(fill=tk.X, pady=(0, 4))
        self.wavelength_nm = tk.StringVar(value="632.8")
        self.w0_mm = tk.StringVar(value="0.80")
        self.p_index = tk.StringVar(value="0")
        self.ell_index = tk.StringVar(value="2")
        self._labeled_entry(beam_frame, "Wavelength [nm]", self.wavelength_nm, 0)
        self._labeled_entry(beam_frame, "Waist w0 [mm]", self.w0_mm, 1)
        self._labeled_entry(beam_frame, "Radial index p", self.p_index, 2)
        self._labeled_entry(beam_frame, "Azimuthal index ell", self.ell_index, 3)

        sim_frame = ttk.LabelFrame(controls, text="Simulation", padding=4)
        sim_frame.pack(fill=tk.X, pady=(0, 4))
        self.grid_size = tk.StringVar(value="512")
        self.window_mm = tk.StringVar(value="8.0")
        self.aperture_mm = tk.StringVar(value="2.5")
        self.propagation_m = tk.StringVar(value="1.0")
        self._labeled_entry(sim_frame, "Grid points N", self.grid_size, 0)
        self._labeled_entry(sim_frame, "Window width [mm]", self.window_mm, 1)
        self._labeled_entry(sim_frame, "Zernike radius [mm]", self.aperture_mm, 2)
        self._labeled_entry(sim_frame, "Propagation z [m]", self.propagation_m, 3)

        z_frame = ttk.LabelFrame(controls, text="Add Zernike aberration", padding=4)
        z_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(z_frame, text="Common aberration").grid(
            row=0, column=0, sticky="w", padx=2, pady=1
        )
        self.preset_var = tk.StringVar(value="Defocus")
        preset_box = ttk.Combobox(
            z_frame, textvariable=self.preset_var, values=list(self.PRESETS),
            state="readonly", width=16,
        )
        preset_box.grid(row=0, column=1, sticky="ew", padx=2, pady=1)
        preset_box.bind("<<ComboboxSelected>>", self._preset_selected)
        self.n_var = tk.StringVar(value="2")
        self.m_var = tk.StringVar(value="0")
        self.weight_var = tk.StringVar(value="0.10")
        self._labeled_entry(z_frame, "Radial order n", self.n_var, 1)
        self._labeled_entry(z_frame, "Azimuthal order m", self.m_var, 2)
        self._labeled_entry(z_frame, "Weight [waves]", self.weight_var, 3)
        ttk.Button(z_frame, text="Add term", command=self.add_term).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=2, pady=(3, 1)
        )

        list_frame = ttk.LabelFrame(controls, text="Active Zernike terms", padding=4)
        list_frame.pack(fill=tk.X, pady=(0, 4))
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.term_tree = ttk.Treeview(
            tree_frame, columns=("n", "m", "weight", "name"),
            show="headings", height=5,
        )
        for column, heading in (("n", "n"), ("m", "m"),
                                ("weight", "Waves"), ("name", "Type")):
            self.term_tree.heading(column, text=heading)
        self.term_tree.column("n", width=26, minwidth=20, anchor="center")
        self.term_tree.column("m", width=26, minwidth=20, anchor="center")
        self.term_tree.column("weight", width=76, minwidth=60, anchor="center")
        self.term_tree.column("name", width=104, minwidth=70, anchor="w")
        tree_scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.term_tree.yview
        )
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.term_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.term_tree.configure(yscrollcommand=tree_scrollbar.set)
        button_row = ttk.Frame(list_frame)
        button_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(button_row, text="Remove selected", command=self.remove_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        ttk.Button(button_row, text="Clear all", command=self.clear_terms).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )
        ttk.Label(
            controls,
            text=("Coefficients: waves of phase.\n"
                  "+m: cosine; -m: sine.\n"
                  "Zernikes apply inside the specified radius."),
            justify=tk.LEFT, wraplength=sidebar_width - 30,
        ).pack(fill=tk.X, pady=(3, 0))
        self._bind_controls_mousewheel(controls)
        self._bind_controls_mousewheel(self.controls_canvas, recursive=False)


    def _resize_controls(self, event):
        """Match the form width to its viewport without horizontal scrolling."""
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)
        self._update_controls_scrollregion()


    def _update_controls_scrollregion(self, event=None):
        bounds = self.controls_canvas.bbox("all")
        if bounds:
            self.controls_canvas.configure(scrollregion=bounds)
            if bounds[3] - bounds[1] <= self.controls_canvas.winfo_height():
                self.controls_canvas.yview_moveto(0)


    def _bind_controls_mousewheel(self, widget, recursive=True):
        # Keep the active-term list's own scrolling behavior.
        if isinstance(widget, (ttk.Treeview, ttk.Scrollbar)):
            return
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._scroll_controls, add="+")
        if recursive:
            for child in widget.winfo_children():
                self._bind_controls_mousewheel(child)


    def _scroll_controls(self, event):
        """Scroll this panel on Windows, macOS, or Linux without global bindings."""
        bounds = self.controls_canvas.bbox("all")
        if not bounds or bounds[3] - bounds[1] <= self.controls_canvas.winfo_height():
            return
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        elif event.delta:
            steps = max(1, abs(int(event.delta)) // 120)
            if event.delta > 0:
                steps = -steps
        else:
            return
        self.controls_canvas.yview_scroll(steps, "units")
        return "break"

    def _build_plot_area(self):
        plot_frame = ttk.Frame(self, padding=(0, 10, 10, 10))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod

    def _labeled_entry(parent, label, variable, row):

        ttk.Label(parent, text=label).grid(

            row=row, column=0, sticky="w", padx=2, pady=1

        )

        entry = ttk.Entry(parent, textvariable=variable, width=10)

        entry.grid(row=row, column=1, sticky="ew", padx=2, pady=1)

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
        validate_zernike_indices(n, m)

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
        parameters = BeamParameters(
            wavelength_nm=float(self.wavelength_nm.get()),
            w0_mm=float(self.w0_mm.get()),
            p=int(self.p_index.get()),
            ell=int(self.ell_index.get()),
            grid_size=int(self.grid_size.get()),
            window_mm=float(self.window_mm.get()),
            aperture_mm=float(self.aperture_mm.get()),
            propagation_m=float(self.propagation_m.get()),
        )
        parameters.validate()
        return parameters

    def update_plots(self):
        try:
            result = simulate_beam(self._read_parameters(), self._read_terms())
            create_plot(result, figure=self.fig)
            self.simulation_result = result
            self.save_button.configure(state=tk.NORMAL)
            self.canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))

    def save_plot_image(self):
        """Save the last successful plotted result, including its settings."""
        if self.simulation_result is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="Save plot image",
            defaultextension=".png",
            initialfile="lg_zernike_plot.png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("TIFF image", "*.tif"),
            ],
        )
        if not filename:
            return
        try:
            path = save_simulation_image(
                self.simulation_result, filename, figure=self.fig, dpi=300
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        messagebox.showinfo("Image saved", f"Saved to:\n{path}")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)  # Per-monitor DPI awareness
        )

    app = LGZernikeApp()
    app.mainloop()