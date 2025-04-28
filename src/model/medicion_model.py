import cv2
import numpy as np
import os
import sys

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class MedicionModel:
    """
    Modelo para la detección y medición de nudos y entrenudos en cañas de azúcar.
    """
    
    def __init__(self):
        # Parámetros del modelo
        self.min_area = 100  # Área mínima para considerar un contorno
        self.kernel_size = 5  # Tamaño del kernel para operaciones morfológicas
        
    def preprocesar_imagen(self, imagen):
        """
        Preprocesa la imagen para mejorar la detección.
        
        Args:
            imagen: Imagen de entrada (numpy array)
        
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        # Convertir a escala de grises si no lo está
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen
        
        # Aplicar filtro gaussiano para reducir ruido
        suavizada = cv2.GaussianBlur(gris, (self.kernel_size, self.kernel_size), 0)
        
        # Aplicar umbral adaptativo
        umbral = cv2.adaptiveThreshold(
            suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Operaciones morfológicas para mejorar la segmentación
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        apertura = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)
        
        return apertura
    
    def detectar_cana(self, imagen_preprocesada):
        """
        Detecta la caña en la imagen preprocesada.
        
        Args:
            imagen_preprocesada: Imagen preprocesada (numpy array)
        
        Returns:
            tuple: (contorno de la caña, máscara de la caña)
        """
        # Encontrar contornos
        contornos, _ = cv2.findContours(
            imagen_preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos pequeños
        contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > self.min_area]
        
        if not contornos_filtrados:
            return None, None
        
        # Encontrar el contorno más grande (asumimos que es la caña)
        contorno_cana = max(contornos_filtrados, key=cv2.contourArea)
        
        # Crear una máscara para la caña
        mascara = np.zeros_like(imagen_preprocesada)
        cv2.drawContours(mascara, [contorno_cana], 0, 255, -1)
        
        return contorno_cana, mascara
    
    def detectar_nudos(self, imagen_original, contorno_cana, mascara):
        """
        Detecta los nudos de la caña utilizando análisis de textura y forma.
        
        Args:
            imagen_original: Imagen original
            contorno_cana: Contorno de la caña
            mascara: Máscara de la caña
        
        Returns:
            list: Lista de contornos de los nudos
        """
        if contorno_cana is None or mascara is None:
            return []
        
        # Convertir a escala de grises si no lo está
        if len(imagen_original.shape) == 3:
            gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen_original
        
        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(gris, gris, mask=mascara)
        
        # Aplicar detector de bordes Canny
        bordes = cv2.Canny(roi, 50, 150)
        
        # Dilatar los bordes para conectar regiones
        kernel = np.ones((3, 3), np.uint8)
        bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
        
        # Encontrar contornos en los bordes
        contornos_nudos, _ = cv2.findContours(
            bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos pequeños
        contornos_nudos = [c for c in contornos_nudos if cv2.contourArea(c) > self.min_area / 2]
        
        # Ordenar los contornos de izquierda a derecha
        x_coords = [cv2.boundingRect(c)[0] for c in contornos_nudos]
        contornos_ordenados = [c for _, c in sorted(zip(x_coords, contornos_nudos))]
        
        return contornos_ordenados
    
    def calcular_medidas(self, imagen, contorno_cana, contornos_nudos, pixeles_por_cm):
        """
        Calcula las medidas de la caña y sus nudos.
        
        Args:
            imagen: Imagen original
            contorno_cana: Contorno de la caña
            contornos_nudos: Lista de contornos de los nudos
            pixeles_por_cm: Factor de conversión de píxeles a cm
        
        Returns:
            dict: Diccionario con las medidas
        """
        if contorno_cana is None:
            return {
                'longitud_total': 0,
                'cantidad_nudos': 0,
                'nudos': [],
                'entrenudos': []
            }
        
        # Obtener el rectángulo que contiene la caña
        x, y, w, h = cv2.boundingRect(contorno_cana)
        
        # Calcular la longitud total en cm
        longitud_total_cm = w / pixeles_por_cm
        
        # Información de los nudos
        nudos = []
        for contorno in contornos_nudos:
            x_nudo, y_nudo, w_nudo, h_nudo = cv2.boundingRect(contorno)
            
            # Posición relativa al inicio de la caña
            posicion_relativa = (x_nudo - x) / pixeles_por_cm
            
            # Ancho del nudo en cm
            ancho_nudo = w_nudo / pixeles_por_cm
            
            nudos.append({
                'posicion': posicion_relativa,
                'ancho': ancho_nudo
            })
        
        # Ordenar los nudos por posición
        nudos = sorted(nudos, key=lambda n: n['posicion'])
        
        # Calcular los entrenudos
        entrenudos = []
        
        # Primer entrenudo (desde el inicio hasta el primer nudo)
        if nudos:
            entrenudos.append({
                'posicion_inicio': 0,
                'posicion_fin': nudos[0]['posicion'] - nudos[0]['ancho'] / 2,
                'longitud': nudos[0]['posicion'] - nudos[0]['ancho'] / 2
            })
            
            # Entrenudos intermedios
            for i in range(len(nudos) - 1):
                inicio = nudos[i]['posicion'] + nudos[i]['ancho'] / 2
                fin = nudos[i + 1]['posicion'] - nudos[i + 1]['ancho'] / 2
                entrenudos.append({
                    'posicion_inicio': inicio,
                    'posicion_fin': fin,
                    'longitud': fin - inicio
                })
            
            # Último entrenudo (desde el último nudo hasta el final)
            entrenudos.append({
                'posicion_inicio': nudos[-1]['posicion'] + nudos[-1]['ancho'] / 2,
                'posicion_fin': longitud_total_cm,
                'longitud': longitud_total_cm - (nudos[-1]['posicion'] + nudos[-1]['ancho'] / 2)
            })
        
        return {
            'longitud_total': round(longitud_total_cm, 2),
            'cantidad_nudos': len(nudos),
            'nudos': [{'posicion': round(n['posicion'], 2), 
                      'ancho': round(n['ancho'], 2)} for n in nudos],
            'entrenudos': [{'longitud': round(e['longitud'], 2)} for e in entrenudos]
        }
    
    def procesar_imagen(self, imagen, referencia_cm=1.64):
        """
        Procesa una imagen para detectar y medir una caña de azúcar.
        
        Args:
            imagen: Imagen de entrada (numpy array)
            referencia_cm: Tamaño de referencia en cm (por defecto 1.64 cm)
        
        Returns:
            tuple: (medidas, imagen_procesada)
        """
        # Calcular la escala (píxeles por cm)
        altura, ancho = imagen.shape[:2]
        pixeles_por_cm = ancho / referencia_cm
        
        # Preprocesar la imagen
        preprocesada = self.preprocesar_imagen(imagen)
        
        # Detectar la caña
        contorno_cana, mascara = self.detectar_cana(preprocesada)
        
        if contorno_cana is None:
            return None, imagen
        
        # Detectar los nudos
        contornos_nudos = self.detectar_nudos(imagen, contorno_cana, mascara)
        
        # Calcular las medidas
        medidas = self.calcular_medidas(imagen, contorno_cana, contornos_nudos, pixeles_por_cm)
        
        # Crear imagen de resultado
        imagen_resultado = imagen.copy()
        
        # Dibujar el contorno de la caña
        cv2.drawContours(imagen_resultado, [contorno_cana], 0, (0, 255, 0), 2)
        
        # Dibujar los contornos de los nudos
        for i, contorno in enumerate(contornos_nudos):
            cv2.drawContours(imagen_resultado, [contorno], 0, (0, 0, 255), 2)
            
            # Añadir etiqueta
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.putText(imagen_resultado, f"Nudo {i+1}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Añadir información de medidas
        cv2.putText(imagen_resultado, f"Longitud: {medidas['longitud_total']} cm", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(imagen_resultado, f"Nudos: {medidas['cantidad_nudos']}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return medidas, imagen_resultado