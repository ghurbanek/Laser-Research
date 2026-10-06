from PIL import Image
import json

with Image.open("training_data_2/abberations/0.png") as image:
    zernike_terms = json.loads(image.info["zernike_terms"])

print(zernike_terms)

# Example: [[2, 0, 0.15], [3, 1, 0.20]]