"""
fast_alignment.py

Finds optimal alignment by a pseudo-gradient descent method. Starting with a manually aligned
as-best-as-can-be-done-by-hand vortex beam (which doesn't have to be very good), user clicks
in center of vortex, then we stochastically walk around the loss surface, adjusting the offset
of the SLM. We do this, occasionally increasing and decreasing the step size to avoid
spurious minima, until a proxy loss function (sum of square differences of brightest value
of intensity slices from the mean brightests) is low enough. Also produces two plots, of the
intensities before and after.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
from sys import exit
from datetime import datetime

import cv2
from screeninfo import get_monitors
from pylablib.devices import uc480

import tkinter as tk
from slm_ui2 import SLMControlsUI

# Setup slm monitor and matrices related to screen size
monitors = get_monitors()
if len(monitors) > 1:
    slm_monitor = monitors[1]
    print("[fast_alignment] Secondary monitor set as SLM display.")
else:
    slm_monitor = monitors[0]
    print("[fast_alignment][DEBUG] SLM not detected. Will display SLM in secondary window on primary monitor.")

H = slm_monitor.width
V = slm_monitor.height

x = np.arange(-H/2, H/2)
y = np.arange(-V/2, V/2)
X, Y = np.meshgrid(x, y)

# Create SLM window
slm_window_name = 'SLM'
if len(monitors) > 1:
    cv2.namedWindow(slm_window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(slm_window_name, slm_monitor.x, slm_monitor.y)
    cv2.setWindowProperty(slm_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow(slm_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slm_window_name, 1200, 900)

# Instantiate camera object, set it to collect grayscale
cam = uc480.UC480Camera()     
cam.set_color_mode("raw8")

# Function to set SLM based on given offset and pre-set variables
def set_slm(x_0, y_0, vals = None):
    """
    Compute hologram and display in SLM window; auto-called by Tkinter.
    Takes optional argument of beam parameter dictionary
    """
    if vals:
        l, nx, ny = vals["l"], vals["nx"], vals["ny"]
    phi = np.angle((X - x_0) + 1j * (Y - y_0))
    gx = nx / H
    gy = ny / V
    hologram = np.mod(l * phi + 2 * np.pi * (Y * gy + X * gx), 2 * np.pi)
    scaled = (hologram / hologram.max() * 255).astype(np.uint8)

    cv2.imshow(slm_window_name, scaled)
    cv2.waitKey(20)

# Create control window and unpack values after pressing "GATHER" button
root = tk.Tk()
ui = SLMControlsUI(root, cam=cam, set_slm_func=set_slm)
root.mainloop()
if hasattr(ui, "values"):
    values = ui.values
    # continue with the other stuff
else:
    print("[fast_alignment] Quitting...")
    exit(0)

# Set center point to the point clicked
x_center, y_center = values["x_center"], values["y_center"]

# Alignment functions
def image_avg(num_frames_to_avg = 5):
    '''
    Take an image using the camera, averaging over five frames to minimize noise.
    '''
    frames_to_avg = []
    _ = cam.grab()[0]     # discard first img grab as camera auto-adjusts
    for _ in range(num_frames_to_avg):
        frame = cam.grab()[0]  # 2D array
        frames_to_avg.append(frame)
    # average frames, and ensure result is a 2D array of 8bit ints
    avg_frame = np.mean(np.stack(frames_to_avg, axis=0), axis=0)
    avg_frame = np.round(avg_frame).astype(np.uint8)
    return avg_frame

def create_intensity_dict(img, x_center, y_center, smooth: bool):
    if smooth:
        img = cv2.blur(img, (10,10))

    center_row = img[y_center]
    center_col = img.T[x_center]

    left_intensity  = center_row[:x_center+1]
    right_intensity = center_row[x_center:]
    up_intensity    = center_col[:y_center+1]
    down_intensity  = center_col[y_center:]

    intensity_dict = {
        "left": left_intensity,
        "right": right_intensity,
        "up": up_intensity,
        "down": down_intensity
    }

    return intensity_dict

def loss_of_image(img):
    # define four intensity profiles
    intensity_dict = create_intensity_dict(img, x_center, y_center, smooth = True)

    # get value of brightest pixel in all four
    brightests = np.zeros(4)
    i=0
    for key in intensity_dict:
        brightests[i] = np.max(intensity_dict[key])
        i += 1

    # proxy loss for those (sum of squared difference for all parameters)
    loss = 4 * np.sum( (brightests - brightests.mean())**2 )

    return loss

def def_offset_map(stepsize):
    offset_map = {
    "A": ( stepsize, 0),
    "B": (-stepsize, 0),
    "C": ( 0, stepsize),
    "D": ( 0,-stepsize)
    }
    return offset_map

# beam imaging functions
def gaussian(x,a,b,c):
        return a * np.exp(-(x - b)**2 / (2*c**2))

def fit_gaussians(intensity_dict):
    '''
    Given the four intensity profiles, return a dictionary giving
    the fitted parameters of each
    '''
    param_dict = {}

    for key in intensity_dict.keys():
        intensity = intensity_dict[key]
        x = np.arange(0, len(intensity),1)
        try:
            params, _ = curve_fit(gaussian, x, intensity)
            param_dict[key] = params
        except RuntimeError:
            param_dict[key] = [1e6, 2e6, 3e6]

    return param_dict

def plot_intensities(img, desired_x, desired_y, x_center, y_center, intensity_dict, param_dict, time_txt, smooth: bool, ui_dir = None):
    '''
    From beam center, go left, right, up, and down to boarder and collect brightnesses
    along the way. Show all four intensity curves and the image with all four of those
    mapped and color-coded.

    Parameters
    ----------
    img : NDArray
        numpy array of 0 - 255 brightness values for image
    desired_x, desired_y : int
        SLM pixel offset
    x_center, y_center : int
        pixel center of beam profile from find_annulus_center function
    intensity_dict : dict
        A dictionary of the four intensity slices, output of 'create_intensity_dict' function
    param_dict : dict
        A dictionary of the Gaussian parameters, output of 'fit_gaussians'
    time_txt : String
        A string saying whether this is pre- or post-alignment.
    smooth : bool
        If true, smooth image before computing intensity profile. Almost always desired.
    ui_dir : str
        If None, will create direction based on file selection from ui. If not None, will be that

    Returns
    -------
    i
    '''
    if smooth:
        img = cv2.blur(img, (10,10)) # smooth over imaged impurities

    # define figure and grid spacing
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 1.6])  

    # 2x2 square plots (take rows 0-3, cols 0-1)
    ax1 = fig.add_subplot(gs[0:2, 0])   # top-left
    ax2 = fig.add_subplot(gs[0:2, 1])   # top-right
    ax3 = fig.add_subplot(gs[2:4, 0])   # bottom-left
    ax4 = fig.add_subplot(gs[2:4, 1])   # bottom-right
    ax_ls = [ax1, ax2, ax3, ax4]

    # Image (span all 4 rows in last col)
    ax_img = fig.add_subplot(gs[:, 2])

    # plot the supposed Gaussians
    direction_ls = ["left","right","up","down"]
    color_ls = ["forestgreen","dodgerblue","red","darkorchid"]
    i = 0
    for ax in [ax1, ax2, ax3, ax4]:
        direction = direction_ls[i]
        x_range = np.arange(0, len(intensity_dict[direction]),1)
        ax.scatter(x_range, intensity_dict[direction], s=6, color=color_ls[i])
        ax.plot(x_range, gaussian(x_range, *param_dict[direction]), color = 'black', linewidth = 3)
        ax.set_ylim(0,265)
        ax.set_box_aspect(1)
        i += 1

    # axis labels for intensity plots
    from matplotlib.transforms import Bbox
    bbox = Bbox.union([ax.get_position() for ax in ax_ls])
    big_ax = fig.add_axes(bbox, frameon=False)
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_xlabel("Pixels", fontsize = 12)
    big_ax.set_ylabel("Intensity", fontsize = 12)

    # show imaged beam
    ax_img.imshow(img, cmap='gray')
    ax_img.axis('off')

    # make colored lines on image
    linewidth = 5
    ax_img.plot([0, x_center], [y_center, y_center], color_ls[0], linestyle='-', linewidth=linewidth)  # left, from (0, y_center) to (x_center, y_center)
    ax_img.plot([x_center, img.shape[1]-1], [y_center, y_center], color_ls[1], linestyle='-', linewidth=linewidth)  # right, from (x_center, y_center) to (img.shape[0], y_center)
    ax_img.plot([x_center, x_center], [0, y_center], color_ls[2], linestyle='-', linewidth=linewidth)  # up, from (x_center, 0) to (x_center, y_center)
    ax_img.plot([x_center, x_center], [y_center, img.shape[0]-1], color_ls[3], linestyle='-', linewidth=linewidth)  # down, from (x_center, y_center) to (x_center, img.shape[1])
    ax_img.set_title(f"Beam profile for x-offset = {desired_x} pixels, y-offset = {desired_y} pixels")

    if smooth:
        smooth_txt = 'smoothed'
    else:
        smooth_txt = 'raw'
    fig.suptitle(f"Axial intensity profiles of {smooth_txt} imaged vortex beam", fontsize = 20)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, f"fig/{ui_dir}")
        
    os.makedirs(fig_dir, exist_ok=True)
    filename = os.path.join(fig_dir, f"intensity_{time_txt}_alignment.png")
    plt.savefig(filename)


# Perform alignment
offset_choices = ["A","B","C","D"]

stepsize = 1
offset_map = def_offset_map(stepsize)

opposite_choice = {
    "A": "B",
    "B": "A",
    "C": "D",
    "D": "C"
}

remaining_choices = offset_choices.copy()
rejects = []

x_off, y_off = 0, 0
set_slm(x_off, y_off, values)  # Set the SLM to initial position
img = image_avg()
L = loss_of_image(img)         # compute loss 

#image beam
intensity_dict = create_intensity_dict(img, x_center, y_center, smooth = True)

param_dict = fit_gaussians(intensity_dict)
plot_intensities(img, 0, 0, x_center, y_center, intensity_dict, param_dict, "before", smooth = True, ui_dir = "test")

# perform alignment loop
while stepsize < 6 and L >= 100:
    # make choice of direction to move (all directions possible first time, after that it changes)
    choice = np.random.choice(remaining_choices)
    
    if len(rejects) == 0 and stepsize == 1:
        # if the last move was accepted, and we've made the choice of next step after
        # disallowing a move in the direction we came, moving in any direction is back in play
        remaining_choices = offset_choices.copy()

    # perform the move
    delta_x, delta_y = offset_map[choice]
    x_off_proposed, y_off_proposed = x_off + delta_x, y_off + delta_y

    # set the slm according to the proposed hologram center
    set_slm(x_off_proposed, y_off_proposed, values) 

    # compute new loss
    img = image_avg()
    L_new = loss_of_image(img)
    print(f"loss: {L_new}", f"x_off: {x_off_proposed}", f"y_off: {y_off_proposed}")

    # accept or reject 
    if L_new < L:
        # accept move
        x_off, y_off = x_off_proposed, y_off_proposed 

        # adjust probabilities so we don't move back the opposite direction and waste compute
        remaining_choices = offset_choices.copy()
        remaining_choices.remove(opposite_choice[choice])

        # reset the list of rejected directions
        rejects = []

        # return to making small steps
        if stepsize != 1:
            stepsize = 1
            offset_map = def_offset_map(stepsize)

        # The set the loss we compare to to be this new one
        L = L_new
    else:
        # reject move (i.e. x_off, y_off = x_off, y_off)

        # adjust probabilities so we don't try this move again
        remaining_choices.remove(choice)
        
        # add attempted move to the list of rejects
        rejects.append(choice)

        # if we've tried every direction, increase the stepsize and try again
        if len(rejects) == 4:
            stepsize += 1
            offset_map = def_offset_map(stepsize)
            remaining_choices = offset_choices.copy()
            rejects = []

# image beam after alignment
set_slm(x_off, y_off, values)  # Set the SLM to initial position
img = image_avg()

intensity_dict = create_intensity_dict(img, x_center, y_center, smooth = True)

param_dict = fit_gaussians(intensity_dict)
plot_intensities(img, x_off, y_off, x_center, y_center, intensity_dict, param_dict, "after", smooth = True, ui_dir = "test")