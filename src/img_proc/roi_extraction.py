import cv2
import numpy as np
from utils import ordenar_puntos

def extraer_roi(image, ancho_max=3840, alto_max=2160):
    """Extrae la región de interés de la imagen."""
    imagen_alineada = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            puntos = ordenar_puntos(approx)
            pts1 = np.float32(puntos)
            rect = cv2.minAreaRect(c)
            ancho_original = int(rect[1][0])
            alto_original = int(rect[1][1])
            ratio = min(ancho_max/ancho_original, alto_max/alto_original)
            ancho_nuevo = int(ancho_original * ratio)
            alto_nuevo = int(alto_original * ratio)
            pts2 = np.float32([[0, 0], [ancho_nuevo, 0], [0, alto_nuevo], [ancho_nuevo, alto_nuevo]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_alineada = cv2.warpPerspective(image, M, (ancho_nuevo, alto_nuevo))
    return imagen_alineada
