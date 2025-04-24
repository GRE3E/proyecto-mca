import cv2
import numpy as np

def preprocesar_imagen_cana(imagen):
    """
    Preprocesa la imagen monocromática de caña de azúcar para unir puntos
    y eliminar imperfecciones.
    """
    # Asegurarse de que la imagen es monocromática
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagen.copy()
        
    # Aplicar umbral para asegurar que es binaria
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invertir si es necesario (asumimos que la caña es blanca sobre fondo negro)
    if np.mean(binary[:10, :10]) > 127:  # Verificar esquina superior izquierda
        binary = cv2.bitwise_not(binary)
    
    # Operaciones morfológicas para cerrar huecos y unir puntos
    # 1. Dilatación para expandir los puntos
    kernel_dilate = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel_dilate, iterations=2)
    
    # 2. Cerrado para unir áreas cercanas
    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Apertura para eliminar pequeños ruidos
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 4. Rellenar huecos internos
    # Primero necesitamos encontrar contornos
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    
    # Filtramos por área para quedarnos con la caña principal
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)  # -1 significa rellenar
    
    # 5. Suavizar bordes
    smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed

def medir_cana_y_nudos(imagen, escala_cm_por_pixel=0.1):
    '''
    Procesa la imagen recortada de la caña para medir:
    - Alto de la caña
    - Cantidad de nudos
    - Cantidad de entre nudos
    - Largo y ancho de cada nudo y entre nudo
    Retorna un diccionario con los resultados.
    '''
    # Primero preprocesamos la imagen para unir los puntos
    imagen_preprocesada = preprocesar_imagen_cana(imagen)
    
    resultado = {
        'alto_cana_px': 0,
        'alto_cana_cm': 0.0,
        'cantidad_nudos': 0,
        'cantidad_entrenudos': 0,
        'nudos': [],  # Lista de {'largo_px', 'ancho_px', 'largo_cm', 'ancho_cm'}
        'entrenudos': []
    }
    
    # Encontrar contornos en la imagen preprocesada
    contours, _ = cv2.findContours(imagen_preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return resultado
    
    # Filtrar contornos por área para eliminar ruido
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    if not contours:
        return resultado
    
    # Suponemos que la caña es el contorno más grande
    caña_contorno = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(caña_contorno)
    
    # Actualizar alto de la caña
    resultado['alto_cana_px'] = h
    if escala_cm_por_pixel:
        resultado['alto_cana_cm'] = h * escala_cm_por_pixel
    
    # Crear una máscara solo con la caña
    mask = np.zeros_like(imagen_preprocesada)
    cv2.drawContours(mask, [caña_contorno], 0, 255, -1)
    
    # Recortar la región de interés de la caña
    caña_roi = imagen_preprocesada[y:y+h, x:x+w]
    
    # Análisis de perfil para detectar nudos
    # Los nudos suelen ser ligeramente más anchos y/o de diferente intensidad
    
    # 1. Análisis de ancho horizontal a diferentes alturas
    anchos = []
    for i in range(0, h, 2):  # Muestrear cada 2 píxeles
        if y+i < mask.shape[0]:
            fila = mask[y+i, x:x+w]
            indices = np.where(fila > 0)[0]
            if len(indices) > 0:
                ancho = indices[-1] - indices[0]
                anchos.append(ancho)
            else:
                anchos.append(0)
    
    # 2. Calcular la primera derivada del perfil de anchos para detectar cambios
    if len(anchos) > 1:
        derivada = np.diff(anchos)
        # Suavizar la derivada
        kernel_size = min(15, len(derivada) // 10 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Asegurar que es impar
        suavizado = cv2.GaussianBlur(np.array(derivada, dtype=np.float32).reshape(-1, 1), 
                                    (1, kernel_size), 0).flatten()
        
        # Detectar picos significativos en la derivada (cambios de ancho)
        umbral_deriv = np.std(suavizado) * 1.5
        picos_indices = []
        
        for i in range(1, len(suavizado)-1):
            if abs(suavizado[i]) > umbral_deriv:
                # Verificar si es un máximo/mínimo local
                if ((suavizado[i] > 0 and suavizado[i] >= suavizado[i-1] and suavizado[i] >= suavizado[i+1]) or
                    (suavizado[i] < 0 and suavizado[i] <= suavizado[i-1] and suavizado[i] <= suavizado[i+1])):
                    picos_indices.append(i)
        
        # Filtrar picos muy cercanos (mismo nudo)
        if picos_indices:
            nudos_posiciones = [picos_indices[0]]
            for idx in picos_indices[1:]:
                if idx - nudos_posiciones[-1] > 15:  # Mínima distancia entre nudos
                    nudos_posiciones.append(idx)
    else:
        nudos_posiciones = []
    
    # Si no hay suficientes nudos detectados, usar método alternativo
    if len(nudos_posiciones) < 3:
        # Análisis de variación de intensidad vertical
        perfil_vertical = np.mean(caña_roi, axis=1)
        perfil_suavizado = cv2.GaussianBlur(perfil_vertical.reshape(-1, 1), (1, 11), 0).flatten()
        
        # Calcular segunda derivada para detectar cambios de curvatura
        derivada1 = np.diff(perfil_suavizado)
        derivada2 = np.diff(derivada1)
        
        # Detectar cruces por cero en la segunda derivada (puntos de inflexión)
        cruces_cero = []
        for i in range(1, len(derivada2)-1):
            if (derivada2[i-1] < 0 and derivada2[i] > 0) or (derivada2[i-1] > 0 and derivada2[i] < 0):
                cruces_cero.append(i)
        
        # Filtrar cruces cercanos
        if cruces_cero:
            nudos_filtrados = [cruces_cero[0]]
            for cruce in cruces_cero[1:]:
                if cruce - nudos_filtrados[-1] > 20:  # Mínima distancia entre nudos
                    nudos_filtrados.append(cruce)
            nudos_posiciones = nudos_filtrados
    
    # Si todavía no hay suficientes nudos, estimar basado en patrones típicos
    if len(nudos_posiciones) < 2:
        # Dividir la caña en secciones aproximadamente iguales
        num_secciones = max(3, h // 40)  # Aproximadamente un nudo cada 40 píxeles
        nudos_posiciones = [i * h // num_secciones for i in range(1, num_secciones)]
    
    # Calcular posiciones reales de los nudos
    nudos_posiciones = [y + pos*2 for pos in nudos_posiciones]  # Multiplicar por 2 porque muestreamos cada 2 píxeles
    
    # Ordenar y filtrar posiciones
    nudos_posiciones.sort()
    
    # Asegurarnos de que no tenemos nudos en los extremos
    if nudos_posiciones and nudos_posiciones[0] < y + 10:
        nudos_posiciones = nudos_posiciones[1:]
    if nudos_posiciones and nudos_posiciones[-1] > y + h - 10:
        nudos_posiciones = nudos_posiciones[:-1]
    
    # Actualizar cantidad de nudos y entrenudos
    resultado['cantidad_nudos'] = len(nudos_posiciones)
    resultado['cantidad_entrenudos'] = max(0, len(nudos_posiciones) - 1)
    
    # Determinar ancho promedio de la caña para estimaciones
    if anchos:
        ancho_promedio = np.mean([a for a in anchos if a > 0])
    else:
        ancho_promedio = w
    
    # Calcular dimensiones de nudos y entrenudos
    for i, pos_y in enumerate(nudos_posiciones):
        # Estimar tamaño del nudo
        nudo_largo = int(h * 0.05)  # 5% de la altura total como estimación
        nudo_largo = max(10, min(nudo_largo, 30))  # Entre 10 y 30 píxeles
        
        # Determinar ancho en la posición del nudo
        pos_y_rel = pos_y - y
        if 0 <= pos_y_rel < len(anchos):
            ancho_nudo = anchos[pos_y_rel]
            if ancho_nudo == 0:  # Si no hay datos en esta posición exacta
                ancho_nudo = ancho_promedio
        else:
            ancho_nudo = ancho_promedio
        
        # Convertir a cm si hay escala
        nudo_largo_cm = nudo_largo * escala_cm_por_pixel if escala_cm_por_pixel else 0.0
        nudo_ancho_cm = ancho_nudo * escala_cm_por_pixel if escala_cm_por_pixel else 0.0
        
        # Agregar información del nudo
        resultado['nudos'].append({
            'largo_px': nudo_largo,
            'ancho_px': int(ancho_nudo),
            'largo_cm': nudo_largo_cm,
            'ancho_cm': nudo_ancho_cm
        })
        
        # Calcular entrenudos (espacio entre nudos)
        if i > 0:
            prev_pos = nudos_posiciones[i-1]
            entrenudo_largo = pos_y - prev_pos - nudo_largo
            # Ajustar si es negativo
            entrenudo_largo = max(5, entrenudo_largo)
            
            # El ancho del entrenudo es el promedio de esta sección
            idx_inicio = (prev_pos - y) // 2
            idx_fin = (pos_y - y) // 2
            if 0 <= idx_inicio < len(anchos) and 0 <= idx_fin < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                if anchos_seccion:
                    ancho_entrenudo = np.mean(anchos_seccion)
                else:
                    ancho_entrenudo = ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
                
            # Convertir a cm
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel if escala_cm_por_pixel else 0.0
            entrenudo_ancho_cm = ancho_entrenudo * escala_cm_por_pixel if escala_cm_por_pixel else 0.0
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': entrenudo_ancho_cm
            })
    
    # Devolver la imagen preprocesada junto con los resultados para visualización
    resultado['imagen_preprocesada'] = imagen_preprocesada
    
    return resultado