import math
import random

from train_model import predict_aberration
from beam import generate_and_save_plot

aberrations : dict[str, list[int]] = {
    "tx" : [1,1],
    "ty" : [1,-1],   
    "ax" : [2,2],
    "ay" : [2,-2],
    "cx" : [3,1],
    "cy" : [3,-1],
    "trx" : [3,3],
    "try" : [3,-3],
}

wv : float = 700
waist : float = 1.0
radial : int = 0
azimuthal : int = 1
grid_sz = 256

def run_model(selected_aberration, selected_strength):
    generate_and_save_plot(
        "test_beam.png",
        wavelength_nm = wv,
        w0_mm = waist,
        p = radial,
        ell = azimuthal,
        propagation_m=1.0,
        grid_size = grid_sz,
        window_mm=8.0,
        zernike_terms=[
            (aberrations.get(selected_aberration)[0], aberrations.get(selected_aberration)[1], selected_strength)
                    ],
        image_type="aberrated",
        folder = None,
    )

    aberration, strength = predict_aberration("test_beam.png")

    print(f"Aberration detected: {aberration} is {selected_aberration == aberration}")
    print(f"Selected strength: {selected_strength}")
    print(f"Strength detected: {math.trunc(strength * 1000) / 1000}")
    print(f"Strength Error: {math.trunc(((selected_strength - strength) / selected_strength) * 100000) / 1000}%")

'''
Tilt X - tx
Tilt Y - ty
Astigmatism X - ax
Astigmatism Y - ay
Coma X - cx
Coma Y - cy
Trefoil X - trx
Trefoil Y -try
'''

if __name__ == "__main__":
    run_model(random.choice(list(aberrations)),random.randint(0,100) / 100)