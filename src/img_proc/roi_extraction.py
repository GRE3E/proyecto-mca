import cv2
import numpy as np
from utils import ordenar_puntos

def extraer_roi(image, ancho_max=3840, alto_max=2160):
    """Extrae la región de interés de la imagen (cuadrado blanco) preservando la relación de aspecto cuadrada."""
    imagen_alineada = None
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbral de Otsu (automáticamente determina el mejor umbral)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # También probar con umbral fijo alto para detectar áreas muy blancas
    _, thresh_high = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Combinar los resultados
    thresh_combined = cv2.bitwise_or(thresh_otsu, thresh_high)
    
    # Operaciones morfológicas para limpiar la imagen
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    cnts = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Si no se encuentran contornos, intentar con el umbral original
    if len(cnts) == 0:
        _, thresh_original = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(thresh_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Filtrar contornos pequeños (menos del 10% del área total)
    area_total = image.shape[0] * image.shape[1]
    cnts = [c for c in cnts if cv2.contourArea(c) > area_total * 0.1]
    
    # Ordenar por área (de mayor a menor)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Buscar el contorno que más se parezca a un cuadrado o rectángulo
    for c in cnts:
        # Calcular perímetro
        peri = cv2.arcLength(c, True)
        
        # Aproximar contorno con diferentes tolerancias
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
            epsilon = epsilon_factor * peri
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            # Si encontramos un cuadrilátero
            if len(approx) == 4:
                # Ordenar puntos
                puntos = ordenar_puntos(approx)
                pts1 = np.float32(puntos)
                
                # IMPORTANTE: Forzar relación de aspecto cuadrada (1:1)
                # Calcular el lado más largo del cuadrilátero
                lado_max = max(
                    np.linalg.norm(pts1[0] - pts1[1]),  # lado superior
                    np.linalg.norm(pts1[2] - pts1[3]),  # lado inferior
                    np.linalg.norm(pts1[0] - pts1[2]),  # lado izquierdo
                    np.linalg.norm(pts1[1] - pts1[3])   # lado derecho
                )
                
                # Usar el lado más largo para crear un cuadrado perfecto
                lado = int(lado_max)
                
                # Transformar perspectiva a un cuadrado perfecto
                pts2 = np.float32([[0, 0], [lado, 0], [0, lado], [lado, lado]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                imagen_alineada = cv2.warpPerspective(image, M, (lado, lado))
                
                return imagen_alineada
    
    # Si no se encontró un cuadrilátero adecuado, intentar con el contorno más grande
    if len(cnts) > 0:
        c = cnts[0]  # El contorno más grande
        
        # Intentar aproximar a un cuadrilátero con una tolerancia mayor
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        # Si tenemos demasiados puntos, usar el rectángulo mínimo
        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            approx = box.reshape(4, 1, 2)
        
        # Ordenar puntos
        puntos = ordenar_puntos(approx)
        pts1 = np.float32(puntos)
        
        # IMPORTANTE: Forzar relación de aspecto cuadrada (1:1)
        # Calcular el lado más largo del cuadrilátero
        lado_max = max(
            np.linalg.norm(pts1[0] - pts1[1]),  # lado superior
            np.linalg.norm(pts1[2] - pts1[3]),  # lado inferior
            np.linalg.norm(pts1[0] - pts1[2]),  # lado izquierdo
            np.linalg.norm(pts1[1] - pts1[3])   # lado derecho
        )
        
        # Usar el lado más largo para crear un cuadrado perfecto
        lado = int(lado_max)
        
        # Transformar perspectiva a un cuadrado perfecto
        pts2 = np.float32([[0, 0], [lado, 0], [0, lado], [lado, lado]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        imagen_alineada = cv2.warpPerspective(image, M, (lado, lado))
    
    return imagen_alineada
