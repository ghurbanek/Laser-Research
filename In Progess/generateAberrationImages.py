from beam import generate_and_save_plot
from pathlib import Path
from matplotlib.image import imsave
from itertools import combinations
import json
from PIL.PngImagePlugin import PngInfo

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
    #"tx" : [1,1],
    #"ty" : [1,-1],   
    "ax" : [2,2],
    "ay" : [2,-2],
    "cx" : [3,1],
    #"cy" : [3,-1],
    #"trx" : [3,3],
    #"try" : [3,-3],
}

#Parameters for ideal beam
wv : float = 700
waist : float = 1.0
radial : int = 0
azimuthal : int = 1
grid_sz = 256


#Generates plot and saves image with metadata that corresponds to the zernike terms
def saveImage(img_name : str, zernike_terms, folder, imageType):
    #gen plot returns (path, image_format, data)
    imageData : tuple = generate_and_save_plot(
        f"{img_name}.png",
        wavelength_nm = wv,
        w0_mm = waist,
        p = radial,
        ell = azimuthal,
        propagation_m=1.0,
        grid_size = grid_sz,
        window_mm=8.0,
        zernike_terms=zernike_terms,
        image_type= imageType,
        folder = folder,
    )

    metadata = PngInfo()
    metadata.add_text("zernike_terms", json.dumps(zernike_terms))

    imsave(
        imageData[0], 
        imageData[2], 
        cmap="gray", 
        origin="lower", 
        vmin=0, 
        vmax=1,
        format=imageData[1],
        pil_kwargs= {"pnginfo" : metadata}
    )


#Finds all possible zernike Combinations Given the dictionary data set
def zernike_combinations(strength : float):
    zernikeCombinations = []
    #R term is how many variables per combination
    for r in range (1, len(aberrations) + 1):
        #Creates a dataset of all possible combinations
        for group in combinations(aberrations, r):
            grouped = []
            #Goes through each aberration in the group and assigns it the correct zernike values

            """
            ADD CREATE DATASET INSIDE HERE TO LOWER PROCESSING TIME
            
            """
            for aberration in group:
                grouped.append([aberrations.get(aberration)[0], aberrations.get(aberration)[1], strength])

            print(group)
            for z in grouped:
                print(z)
            zernikeCombinations.append(grouped)
            
    return zernikeCombinations


#Start and Stop are inclusive
def create_dataset(strength : float):
    #Creates Data Set Folder
    data = Path("training_data_2")
    data.mkdir(exist_ok=True)

    #Ideal Beam, Creation
    saveImage("ideal", [], data, "ideal")

    #Creates Sub Folder
    nonIdeal = Path(data / f"abberations")
    nonIdeal.mkdir(exist_ok=True)
    
    #Finds all possible zernikeCombinations at a given strength
    zernikeCombinations = zernike_combinations(.5)

    z = 0
    while(z < len(zernikeCombinations)):
        saveImage(str(z), zernikeCombinations[z], nonIdeal, "aberrated")
        z += 1


if __name__ == "__main__":
    create_dataset(.5)

#Iterate Through Multiple Strengths