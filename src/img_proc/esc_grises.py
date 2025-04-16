import cv2
import numpy as np

def convertir_a_grises(imagen):
    
    if len(imagen.shape) == 3:  # Si la imagen está en color
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen  # Si ya está en escala de grises, la devolvemos sin cambios