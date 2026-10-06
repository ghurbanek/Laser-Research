"""
slm_ui2.py

Contains the UI for setting specific parameters of the SLM hologram (l, nx, ny),
viewing the beam with the camera, and performing alignment. Alignment is done by clicking
on the center of a best-guess manually aligned beam with the mouse, and then perturbing
the SLM to symmetrize the four axial intensity profiles.
"""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2

class SLMControlsUI():
    def __init__(self, master, cam=None, set_slm_func=None):
        """
        cam : optional camera object passed from main script
        set_slm_func : optional function handle (x0,y0,l,nx,ny) to refresh SLM
        """
        self.master = master
        self.cam = cam
        self.set_slm_func = set_slm_func

        # window setup
        master.title("SLM Controls")
        frame_top = ttk.Frame(master)
        frame_top.pack(padx=10, pady=5, fill="x")

        # box and label for topological charge
        ttk.Label(frame_top, text="Topological Charge (l)", justify="center").pack()    # make text box
        self.top_charge = tk.StringVar(value="2")                                       # default value
        self.top_charge_entry = ttk.Entry(frame_top, width=20, textvariable = self.top_charge)  # variable from text entry
        self.top_charge_entry.bind("<Return>", lambda e: self.update_slm())      # pressing 'enter' updates SLM
        self.top_charge_entry.bind("<FocusOut>", lambda e: self.update_slm())    # clicking out of box updates
        self.top_charge_entry.pack(pady=2)                                                      # make pretty

        # box and label for n_x
        ttk.Label(frame_top, text="Horizontal Grooves (nₓ)", justify="center").pack()
        self.nx = tk.StringVar(value="150")
        self.nx_entry = ttk.Entry(frame_top, width=20, textvariable = self.nx)
        self.nx_entry.bind("<Return>", lambda e: self.update_slm())
        self.nx_entry.bind("<FocusOut>", lambda e: self.update_slm())
        self.nx_entry.pack(pady=2)

        # box and label for n_y
        ttk.Label(frame_top, text="Vertical Grooves (nᵧ)", justify="center").pack()
        self.ny = tk.StringVar(value="150")
        self.ny_entry = ttk.Entry(frame_top, width=20, textvariable = self.ny)
        self.nx_entry.bind("<Return>", lambda e: self.update_slm())
        self.nx_entry.bind("<FocusOut>", lambda e: self.update_slm())
        self.ny_entry.pack(pady=2)

        # initial set of beam center variables so program doesn't freak it when they aren't defined
        self.beam_center_x = 500
        self.beam_center_y = 500

        # buttons to quit, preview, and start data aligning
        frame_buttons = ttk.Frame(master)
        frame_buttons.pack(pady=10)

        quit_btn = tk.Button(frame_buttons, text="QUIT", bg="red", fg="white", command=master.destroy)
        quit_btn.grid(row=0, column=0, padx=15)

        preview_btn = tk.Button(frame_buttons, text="PREVIEW", bg="blue", fg="white", command=self.preview_button)
        preview_btn.grid(row=0, column=1, padx=15)

        self.align_clicked = False
        align_btn = tk.Button(frame_buttons, text="ALIGN", bg="green", fg="white", command=self.align_button)
        align_btn.grid(row=0, column=2, padx=15)

        if self.set_slm_func:
            vals = self.get_values()
            self.set_slm_func(0, 0, vals)  # center hologram at (0,0) during preview

    def update_slm(self, *args):
        if self.set_slm_func:
            vals = self.get_values()
            self.set_slm_func(0, 0, vals)

    def preview_button(self):
        self.update_slm()
        if self.cam is None:
            print("[slm_ui2] No camera passed to UI, cannot preview.")
            return
        print("[slm_ui2] Entering preview mode (press 'q' in window to quit).")

        vals = self.get_values()
        if self.set_slm_func:
            self.set_slm_func(0, 0, vals)  # center hologram at (0,0) during preview

        while True:
            frame = self.cam.grab()[0]

            # check for saturation: more than 64 pixels with value 255
            if (frame >= 255).sum() > 64:
                cv2.putText(frame,
                            "WARNING: Several pixels capped at 255 brightness",
                            (10, 30),                      # position (x,y)
                            cv2.FONT_HERSHEY_SIMPLEX,      # font
                            0.7,                           # font scale
                            (0, 0, 255),                   # text color (red)
                            2,                             # thickness
                            cv2.LINE_AA)
            
            cv2.putText(frame,
                        "Press 'q' to close preview.",
                        (10, len(frame) - 30),                      # position (x,y)
                        cv2.FONT_HERSHEY_SIMPLEX,      # font
                        0.7,                           # font scale
                        (255, 255, 255),                   # text color (red)
                        2,                             # thickness
                        cv2.LINE_AA)

            cv2.imshow("Live Preview", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Live Preview")

    def align_button(self):
        if self.cam is None:
            print("[slm_ui2] No camera passed to UI, cannot align.")
            return

        print("[slm_ui2] Entering alignment mode.")
        print("Please click on the beam center in the preview window.")

        # Update SLM before alignment
        self.update_slm()
        vals = self.get_values()
        if self.set_slm_func:
            self.set_slm_func(0, 0, vals)

        clicked_point = [None]
        confirmed = False

        # Create a simple Tkinter root (for message boxes)
        root = tk.Tk()
        root.withdraw()  # hide main Tk window

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)

                # Show the dot right away
                frame = self.cam.grab()[0]
                display = frame.copy()
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Align Beam", display)
                cv2.waitKey(1)  # force refresh before popup appears

                # Ask user for confirmation via popup
                answer = messagebox.askyesno(
                    "Confirm Alignment",
                    f"Is this the correct beam center?\n(x, y) = ({x}, {y})"
                )
                if answer:
                    nonlocal confirmed
                    confirmed = True
                else:
                    clicked_point[0] = None

        cv2.namedWindow("Align Beam")
        cv2.setMouseCallback("Align Beam", click_event)

        while True:
            frame = self.cam.grab()[0]
            display = frame.copy()

            # Draw circle on clicked point (if exists)
            if clicked_point[0] is not None:
                cv2.circle(display, clicked_point[0], 5, (0, 0, 255), -1)

            cv2.imshow("Align Beam", display)
            key = cv2.waitKey(1) & 0xFF

            if confirmed:
                break
            elif key == ord('q'):
                break

        cv2.destroyWindow("Align Beam")
        root.destroy()

        if confirmed and clicked_point[0] is not None:
            self.beam_center_x, self.beam_center_y = clicked_point[0]
            print(f"[slm_ui2] Beam center set to {self.beam_center_x}, {self.beam_center_y}")
        else:
            print("[slm_ui2] Alignment cancelled or not confirmed.")

        self.values = self.get_values()
        # close the window
        self.master.destroy()

        
    def get_values(self):
        ui_value_dict = {
            "l" : int(self.top_charge.get()),
            "nx" : int(self.nx.get()),
            "ny" : int(self.ny.get()),

            "x_center" : int(self.beam_center_x),
            "y_center" : int(self.beam_center_y),

            "align_clicked" : self.align_clicked
        }
        return ui_value_dict


if __name__ == "__main__":
    root = tk.Tk()
    app = SLMControlsUI(root)
    root.mainloop()
