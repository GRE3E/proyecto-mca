import cv2
import numpy as np
from PIL import Image
from utils import ordenar_puntos

class ImageProcessor:
    """Clase para el procesamiento de imágenes."""
    
    def roi(self, image, ancho_max=1920, alto_max=1080):
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
                
                # Calcular el ancho y alto manteniendo la relación de aspecto
                rect = cv2.minAreaRect(c)
                ancho_original = int(rect[1][0])
                alto_original = int(rect[1][1])
                
                # Mantener la relación de aspecto
                ratio = min(ancho_max/ancho_original, alto_max/alto_original)
                ancho_nuevo = int(ancho_original * ratio)
                alto_nuevo = int(alto_original * ratio)
                
                pts2 = np.float32([[0, 0], [ancho_nuevo, 0], [0, alto_nuevo], [ancho_nuevo, alto_nuevo]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                imagen_alineada = cv2.warpPerspective(image, M, (ancho_nuevo, alto_nuevo))
        return imagen_alineada
    
    def detectar_bordes(self, imagen):
        """Detecta bordes en la imagen usando procesamiento de NumPy."""
        # Convertir la imagen recortada a array de NumPy
        img_array = np.array(Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)))
        
        # Crear una copia para el procesamiento de bordes
        output_array = img_array.copy()
        
        # Calcular la luminosidad promedio
        luminosity = np.mean(img_array, axis=2)
        
        # Calcular diferencias manteniendo dimensiones consistentes
        diff_x = np.zeros_like(luminosity)
        diff_y = np.zeros_like(luminosity)
        diff_x[:, :-1] = np.abs(luminosity[:, :-1] - luminosity[:, 1:])
        diff_y[:-1, :] = np.abs(luminosity[:-1, :] - luminosity[1:, :])
        
        # Crear máscara de bordes
        threshold = 10
        edge_mask = (diff_x + diff_y) < threshold
        
        # Aplicar máscara
        output_array[edge_mask] = 0
        output_array[~edge_mask] = np.clip(luminosity[~edge_mask].astype(int) - 1, 0, 255)[:, None]
        
        # Convertir array procesado a imagen PIL
        pil_image = Image.fromarray(output_array.astype('uint8'))
        
        # Liberar memoria
        del img_array
        del output_array
        del luminosity
        del diff_x
        del diff_y
        
        return pil_image
    
    def calcular_mediciones(self, imagen_bordes, imagen_original):
        """Calcula mediciones basadas en contornos detectados."""
        # Convertir PIL a OpenCV para trabajar con contornos
        imagen_bordes_cv = cv2.cvtColor(np.array(imagen_bordes), cv2.COLOR_RGB2BGR)
        imagen_bordes_gray = cv2.cvtColor(imagen_bordes_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imagen_bordes_gray, 127, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen en blanco para las mediciones
        imagen_mediciones = np.zeros_like(imagen_original)
        
        metros_por_pixel = 0
        x_ref, y_ref, w_ref, h_ref = 0, 0, 0, 0
        
        # Encontrar el cuadrado de referencia (asumimos que es el contorno más grande)
        if contornos:
            # Ordenar contornos por área
            contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
            
            # El primer contorno debería ser el cuadrado de referencia
            cnt_referencia = contornos[0]
            x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(cnt_referencia)
            
            # Calcular la relación metros/píxeles usando el cuadrado de 1.64m
            metros_por_pixel = 1.64 / max(w_ref, h_ref)
            
            # Dibujar y medir cada contorno en la imagen de bordes
            for cnt in contornos:
                self._dibujar_mediciones(imagen_bordes_cv, cnt, metros_por_pixel)
            
            # Marcar el cuadrado de referencia
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(imagen_bordes_cv, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (0, 0, 255), 2)
            cv2.putText(imagen_bordes_cv, 'Referencia (1.64m)', (x_ref, y_ref-10), font, 0.5, (0, 0, 255), 1)
            
            # Dibujar mediciones en imagen en blanco
            font = cv2.FONT_HERSHEY_SIMPLEX
            for cnt in contornos:
                self._dibujar_mediciones(imagen_mediciones, cnt, metros_por_pixel, color=(255, 255, 255))
            
            # Marcar referencia en imagen de mediciones
            cv2.rectangle(imagen_mediciones, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (255, 255, 255), 2)
            cv2.putText(imagen_mediciones, 'Referencia (1.64m)', (x_ref, y_ref-10), font, 0.5, (255, 255, 255), 1)
        
        return imagen_bordes_cv, imagen_mediciones
    
    def _dibujar_mediciones(self, imagen, contorno, metros_por_pixel, color=(0, 255, 0)):
        """Dibuja las mediciones en la imagen."""
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Calcular dimensiones en metros
        ancho_m = w * metros_por_pixel
        alto_m = h * metros_por_pixel
        
        # Dibujar rectángulo
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 2)
        
        # Dibujar líneas de medición
        cv2.line(imagen, (x, y + h//2), (x + w, y + h//2), (255, 0, 0) if color != (255, 255, 255) else color, 1)
        cv2.line(imagen, (x + w//2, y), (x + w//2, y + h), (255, 0, 0) if color != (255, 255, 255) else color, 1)
        
        # Mostrar medidas
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imagen, f'{ancho_m:.2f}m', (x, y-5), font, 0.5, color, 1)
        cv2.putText(imagen, f'{alto_m:.2f}m', (x+w+5, y+h//2), font, 0.5, color, 1)
    
    def redimensionar_imagen(self, imagen):
        """Redimensiona la imagen para procesamiento manteniendo la relación de aspecto."""
        max_dimension = 800
        height, width = imagen.shape[:2]
        scale = min(max_dimension/width, max_dimension/height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            imagen = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return imagen
    
    def procesar_imagen_completa(self, ruta_imagen):
        """Procesa la imagen completa y devuelve los resultados."""
        # Leer imagen con OpenCV
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            raise ValueError("No se pudo leer la imagen")
        
        # Redimensionar para procesamiento
        imagen = self.redimensionar_imagen(imagen)
        
        # Extraer región de interés
        imagen_A4 = self.roi(imagen)
        if imagen_A4 is None:
            imagen_A4 = imagen.copy()
        
        # Detectar bordes
        pil_image = self.detectar_bordes(imagen_A4)
        
        # Calcular mediciones
        imagen_bordes, imagen_mediciones = self.calcular_mediciones(pil_image, imagen_A4)
        
        return {
            'imagen_A4': imagen_A4,
            'imagen_bordes': pil_image,
            'imagen_mediciones': imagen_mediciones
        }