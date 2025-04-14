import numpy as np
from PIL import Image
import cv2

def detectar_bordes(imagen):
    """Detecta bordes en la imagen usando procesamiento de NumPy."""
    imagen_bgr = cv2.GaussianBlur(imagen, (5, 5), 0) # Desenfoque Gaussiano
    img_array = np.array(Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)))
    output_array = img_array.copy()
    luminosity = np.mean(img_array, axis=2)
    diff_x = np.zeros_like(luminosity)
    diff_y = np.zeros_like(luminosity)
    diff_x[:, :-1] = np.abs(luminosity[:, :-1] - luminosity[:, 1:])
    diff_y[:-1, :] = np.abs(luminosity[:-1, :] - luminosity[1:, :])
    threshold = 40 #25 (original 10)(si sirve 40 pero desaparece parte de la imagen principal)
    edge_mask = (diff_x + diff_y) < threshold
    output_array[edge_mask] = 0
    output_array[~edge_mask] = np.clip(luminosity[~edge_mask].astype(int) - 1, 0, 255)[:, None]
    pil_image = Image.fromarray(output_array.astype('uint8'))
    
    return pil_image