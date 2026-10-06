"""
vortex_interactive_optimized_enter.py

Optimized SLM hologram generator:
- Precomputed Zernike basis (Noll 2–11)
- Cached geometry
- Cached phases
- Linear combination architecture
- CPU only
- Enter-key update model (no live recompute while typing)
"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from screeninfo import get_monitors

# =========================
# ZERNIKE DEFINITIONS
# =========================

def fact(n):
    n = int(n)
    prod = 1
    for i in range(1, n+1):
        prod *= i
    return prod

def N(n,m):
    delta_m0 = 1 if m == 0 else 0
    return np.sqrt(2*(n+1)/(1+delta_m0))

def R(n,m,rho):
    R_nm = 0
    for s in range(int((n-np.abs(m))/2)+1):
        num = (-1)**s * fact(n-s) * rho**(n-2*s)
        den = fact(s)*fact((n+np.abs(m))/2 - s)*fact((n-np.abs(m))/2 - s)
        R_nm += num/den
    return R_nm

def Z(n,m,rho,theta,pupil):
    if m >= 0:
        return N(n,m)*R(n,m,rho)*np.cos(m*theta)*pupil
    else:
        return -N(n,m)*R(n,m,rho)*np.sin(m*theta)*pupil

# Noll map
noll = {
    2:(1,1),
    3:(1,-1),
    4:(2,0),
    5:(2,-2),
    6:(2,2),
    7:(3,-1),
    8:(3,1),
    9:(3,-3),
    10:(3,3),
    11:(4,0)
}

noll_names = {
    2: "Tilt x",
    3: "Tilt y",
    4: "Defocus",
    5: "Astigmatism 45",
    6: "Astigmatism 0",
    7: "Coma y",
    8: "Coma x",
    9: "Trefoil y",
    10:"Trefoil x",
    11:"Spherical"
}

# =========================
# MONITOR SETUP
# =========================

monitors = get_monitors()
if len(monitors) > 1:
    slm_monitor = monitors[1]
else:
    slm_monitor = monitors[0]

H = slm_monitor.width
V = slm_monitor.height

x = np.arange(-H/2, H/2)
y = np.arange(-V/2, V/2)
X, Y = np.meshgrid(x, y)

# =========================
# SLM WINDOW
# =========================

slm_window_name = 'SLM'
if len(monitors) > 1:
    cv2.namedWindow(slm_window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(slm_window_name, slm_monitor.x, slm_monitor.y)
    cv2.setWindowProperty(slm_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow(slm_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slm_window_name, 1200, 900)

# =========================
# TKINTER UI
# =========================

root = tk.Tk()
root.title("SLM Controls")
root.geometry("520x760")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

page_main = ttk.Frame(notebook)
page_aberr = ttk.Frame(notebook)

notebook.add(page_main, text="Vortex + Grating")
notebook.add(page_aberr, text="Aberrations")

def labeled_entry(parent, label, default):
    tk.Label(parent, text=label).pack()
    e = tk.Entry(parent)
    e.insert(0, str(default))
    e.pack()
    return e

# Main controls
tk.Label(page_main, text="Vortex and Grating Controls", font=("Arial", 12, "bold")).pack(pady=10)
entry_l  = labeled_entry(page_main, "Topological Charge (l)", 2)
entry_nx = labeled_entry(page_main, "X Groove Count (n_x)", 50)
entry_ny = labeled_entry(page_main, "Y Groove Count (n_y)", 50)
entry_x0 = labeled_entry(page_main, "Horizontal Center Shift (x0)", 0)
entry_y0 = labeled_entry(page_main, "Vertical Center Shift (y0)", 0)

tk.Label(page_main, text="Press 'Enter' to Update Hologram", font=("Arial", 10, "bold")).pack(pady=50)

# Aberration
tk.Label(page_aberr, text="Zernike Aberrations (Noll 2–11)", font=("Arial", 12, "bold")).pack(pady=10)

aberr_entries = []
for idx in range(2,12):
    e = labeled_entry(page_aberr, f"{noll_names[idx]} (Noll {idx})", 0.0)
    aberr_entries.append(e)

entry_pupil = labeled_entry(page_aberr, "Pupil radius fraction (0–1)", 0.5)

# =========================
# SAFE PARSING
# =========================

def safe_int(entry, default):
    try: return int(entry.get())
    except: return default

def safe_float(entry, default):
    try: return float(entry.get())
    except: return default

# =========================
# PRECOMPUTATION STORAGE
# =========================

Theta_base = np.arctan2(Y, X)
phi_grating_base = np.zeros_like(X)
Z_cache = [None]*10
pupil_mask = None
Rho = None
Theta = 1

# =========================
# PRECOMPUTE FUNCTIONS
# =========================

def precompute_geometry(pupil_frac, x0, y0):
    global Rho, Theta, pupil_mask, Z_cache

    Rho_scale = min(H,V)/2 * pupil_frac
    Rho = np.sqrt((X-x0)**2 + (Y-y0)**2) / Rho_scale
    Theta = np.arctan2(Y-y0, X-x0)
    pupil_mask = (Rho <= 1)

    for i in range(10):
        n,m = noll[i+2]
        Z_cache[i] = Z(n,m,Rho,Theta,pupil_mask)

def precompute_grating(nx, ny):
    global phi_grating_base
    gx = nx / H
    gy = ny / V
    phi_grating_base = 2*np.pi*(X*gx + Y*gy)

# =========================
# PARAM READERS
# =========================

def read_main():
    return (
        safe_int(entry_l,2),
        safe_int(entry_nx,50),
        safe_int(entry_ny,50),
        safe_int(entry_x0,0),
        safe_int(entry_y0,0)
    )

def read_aberr():
    a = []
    for e in aberr_entries:
        a.append(safe_float(e,0))
    pupil_frac = safe_float(entry_pupil,0.5)
    return np.array(a), pupil_frac

# =========================
# UPDATE FUNCTION
# =========================

def update_hologram():
    l,nx,ny,x0,y0 = read_main()
    a, pupil_frac = read_aberr()

    precompute_geometry(pupil_frac, x0, y0)
    precompute_grating(nx, ny)

    phi_vortex = l * Theta
    phi_aberr = np.zeros_like(X)

    for i in range(10):
        phi_aberr += a[i] * Z_cache[i]

    hologram = np.mod(phi_vortex + phi_grating_base + phi_aberr, 2*np.pi)
    scaled = (hologram/(2*np.pi)*255).astype(np.uint8)

    cv2.imshow(slm_window_name, scaled)
    cv2.waitKey(1)

# =========================
# ENTER-ONLY BINDING
# =========================

def bind_entry_enter(entry):
    entry.bind("<Return>", lambda e: update_hologram())

all_entries = [entry_l,entry_nx,entry_ny,entry_x0,entry_y0,entry_pupil] + aberr_entries
for e in all_entries:
    bind_entry_enter(e)

# =========================
# EXIT
# =========================

def quit_program():
    cv2.destroyAllWindows()
    root.destroy()

tk.Button(root, text="Quit", command=quit_program, bg='red', fg='white').pack(pady=8)

# =========================
# INIT
# =========================

update_hologram()
root.mainloop()