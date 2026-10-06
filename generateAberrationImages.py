from beam import generate_and_save_plot
from pathlib import Path

#Aberrations shorthand conversion
"""
Piston - ps
Tilt X - tx
Tilt Y - ty
Defocus - df
Astigmatism X - ax
Astigmatism Y - ay
Coma X - cx
Coma Y - cy
Trefoil X - trx
Trefoil Y -try
Primary Spherical - prs
"""
#Possible Aberrations
#Piston, Defocus, Primary Spherical is excluded
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

#Parameters for ideal beam
wv : float = 700
waist : float = 1.0
radial : int = 0
azimuthal : int = 1
grid_sz = 256

#Start and Stop are inclusive
def create_dataset(start : int, stop : int, step : float):
    #Creates Data Set Folder
    data = Path("training_data")
    data.mkdir(exist_ok=True)

    #Ideal Bea, Creatopm
    generate_and_save_plot(
            "ideal.png",
            wavelength_nm = wv,
            w0_mm = waist,
            p = radial,
            ell = azimuthal,
            propagation_m=1.0,
            grid_size = grid_sz,
            window_mm=8.0,
            image_type="ideal",
            folder = data,
        )

    #Variable strength of aberration
    for aberration in aberrations:

        #Creates Sub Folder
        abVariable = Path(data / f"training_{aberration}")
        abVariable.mkdir(exist_ok=True)

        strength : float = start
        while(strength <= stop):
            generate_and_save_plot(
                        f"{aberration}_{strength}.png",
                        wavelength_nm = wv,
                        w0_mm = waist,
                        p = radial,
                        ell = azimuthal,
                        propagation_m=1.0,
                        grid_size = grid_sz,
                        window_mm=8.0,
                        zernike_terms=[
                            (aberrations.get(aberration)[0], aberrations.get(aberration)[1], strength)
                                    ],
                        image_type="aberrated",
                        folder = abVariable,
                    )
            strength += step

if __name__ == "__main__":
    create_dataset(0,1,.001)
    print("Finished")