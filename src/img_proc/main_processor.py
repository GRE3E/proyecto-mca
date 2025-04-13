import cv2
from .roi_extraction import extraer_roi
from .edge_detection import detectar_bordes
from .measurements import calcular_mediciones

class ImageProcessor:
    def redimensionar_imagen(self, imagen):
        return imagen
    
    def procesar_imagen_completa(self, ruta_imagen):
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            raise ValueError("No se pudo leer la imagen")
        
        imagen = self.redimensionar_imagen(imagen)
        imagen_A4 = extraer_roi(imagen)
        if imagen_A4 is None:
            imagen_A4 = imagen.copy()
        
        pil_image = detectar_bordes(imagen_A4)
        imagen_bordes, imagen_mediciones = calcular_mediciones(pil_image, imagen_A4)
        
        return {
            'imagen_A4': imagen_A4,
            'imagen_bordes': pil_image,
            'imagen_mediciones': imagen_mediciones
        }