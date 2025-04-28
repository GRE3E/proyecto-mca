import cv2
from .roi_extraction import extraer_roi
from .edge_detection import detectar_bordes

class ImageProcessor:
    def redimensionar_imagen(self, imagen):
        import cv2
        return cv2.resize(imagen, (1640, 1640), interpolation=cv2.INTER_AREA)
    
    def procesar_imagen_completa(self, ruta_imagen):
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            raise ValueError("No se pudo leer la imagen")
        
        imagen = self.redimensionar_imagen(imagen)
        imagen_A4 = extraer_roi(imagen)
        if imagen_A4 is None:
            imagen_A4 = imagen.copy()
        
        imagen_bordes = detectar_bordes(imagen_A4)
        
        return {
            'imagen_A4': imagen_A4,
            'imagen_bordes': imagen_bordes
        }