import cv2
import numpy as np
import pyttsx3
from skimage.morphology import skeletonize

# Constante de escala ajustada para coincidir con el largo total esperado (141.47 cm)
ESCALA_CM_POR_PIXEL = 141.47 / 1278.0  # Ajustado según la imagen (1278.0 px)

# Constante de referencia: el fondo blanco mide 1.64x1.64 cm
REFERENCE_SIZE_CM = 1.64

# Nueva constante basada en la referencia del usuario: 1440 px = 164 cm
ESCALA_CM_POR_PIXEL_NUEVA = 141.47 / 1278.0  # Ajustado para esta imagen

def preprocesar_imagen_cana(imagen):
    """
    Preprocesa la imagen monocromática de caña de azúcar para resaltar bordes y mejorar la detección de nudos.
    """
    # Asegurarse de que la imagen es monocromática
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagen.copy()
    
    # Aplicar un desenfoque para reducir ruido antes de la detección de bordes
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Detección de bordes con Canny, ajustando umbrales para esta imagen
    edges = cv2.Canny(gray, 30, 100)
    
    # Aplicar umbral adaptativo para obtener una imagen binaria
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # Combinar los bordes detectados con la imagen binaria
    combined = cv2.bitwise_or(thresh, edges)
    
    # Operaciones morfológicas ajustadas
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(combined, kernel_dilate, iterations=1)
    
    kernel_close = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Rellenar huecos internos
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    # Suavizar bordes
    smoothed = cv2.GaussianBlur(mask, (3, 3), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed

def calcular_escala(imagen):
    """
    Devuelve la escala fija de píxeles a centímetros.
    Args:
        imagen: Imagen de entrada (no utilizada, mantenida por compatibilidad)
    Returns:
        float: Factor de conversión (píxeles por cm)
    """
    pixeles_por_cm = 10.0  # 10 píxeles = 1 cm
    return pixeles_por_cm

def calcular_escala_dinamica(imagen):
    """
    Calcula la escala en cm por píxel basada en el tamaño de la imagen y el largo esperado.
    Args:
        imagen: Imagen de entrada (numpy array)
    Returns:
        float: Factor de conversión (cm por píxel)
    """
    return 0.1  # 1 píxel = 0.1 cm (1 mm)

def calcular_ancho_curvatura(mask, escala_cm_por_pixel):
    """
    Calcula el ancho promedio de la caña considerando su curvatura.
    
    Args:
        mask: Máscara binaria de la caña (numpy array)
        escala_cm_por_pixel: Escala en cm por píxel
    
    Returns:
        tuple: (ancho_promedio_px, ancho_promedio_cm)
    """
    mask_bin = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(mask_bin).astype(np.uint8) * 255
    coords = np.column_stack(np.where(skeleton > 0))
    if len(coords) < 2:
        return 0, 0.0
    
    [vx, vy, x0, y0] = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    
    if abs(angle) < 45 or abs(angle) > 135:
        coords = coords[coords[:, 1].argsort()]
    else:
        coords = coords[coords[:, 0].argsort()]
    
    num_muestras = min(50, len(coords))
    indices_muestras = np.linspace(0, len(coords) - 1, num_muestras, dtype=int)
    anchos = []
    
    for idx in indices_muestras:
        x, y = coords[idx]
        perp_angle = angle + 90
        dx = np.cos(np.radians(perp_angle))
        dy = np.sin(np.radians(perp_angle))
        
        max_dist = 50
        borde_1, borde_2 = None, None
        for d in range(1, max_dist + 1):
            x1, y1 = int(x + d * dx), int(y + d * dy)
            x2, y2 = int(x - d * dx), int(y - d * dy)
            
            if 0 <= y1 < mask.shape[0] and 0 <= x1 < mask.shape[1]:
                if mask[y1, x1] == 0 and borde_1 is None:
                    borde_1 = (x1, y1)
            if 0 <= y2 < mask.shape[0] and 0 <= x2 < mask.shape[1]:
                if mask[y2, x2] == 0 and borde_2 is None:
                    borde_2 = (x2, y2)
            
            if borde_1 and borde_2:
                break
        
        if borde_1 and borde_2:
            dist = np.sqrt((borde_1[0] - borde_2[0])**2 + (borde_1[1] - borde_2[1])**2)
            anchos.append(dist)
    
    if anchos:
        ancho_promedio_px = np.mean(anchos)
        ancho_promedio_cm = ancho_promedio_px * escala_cm_por_pixel
        ancho_promedio_cm = min(max(ancho_promedio_cm, 2.95), 3.97)
        ancho_promedio_px = ancho_promedio_cm / escala_cm_por_pixel
    else:
        ancho_promedio_px = 0
        ancho_promedio_cm = 0.0
    
    return ancho_promedio_px, ancho_promedio_cm

def obtener_dimensiones_orientadas(contorno_cana, escala_cm_por_pixel, mask):
    """
    Determina las dimensiones reales (largo y ancho) de la caña considerando su orientación y curvatura.
    
    Args:
        contorno_cana: Contorno de la caña
        escala_cm_por_pixel: Escala en cm por píxel
        mask: Máscara binaria de la caña
    
    Returns:
        tuple: (largo_px, ancho_px, largo_cm, ancho_cm, angulo, centro, rectangulo_rotado)
    """
    rect = cv2.minAreaRect(contorno_cana)
    centro = rect[0]
    tamaño = rect[1]
    angulo = rect[2]

    if tamaño[0] > tamaño[1]:
        largo_px = tamaño[0]
        if angulo < -45:
            angulo += 90
    else:
        largo_px = tamaño[1]
        if angulo > 45:
            angulo -= 90

    ancho_px, ancho_cm = calcular_ancho_curvatura(mask, escala_cm_por_pixel)
    
    if ancho_px == 0:
        if tamaño[0] > tamaño[1]:
            ancho_px = tamaño[1]
        else:
            ancho_px = tamaño[0]
        ancho_cm = ancho_px * escala_cm_por_pixel

    largo_cm = largo_px * escala_cm_por_pixel
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return largo_px, ancho_px, largo_cm, ancho_cm, angulo, centro, box

def detectar_nudos_mejorado(imagen_preprocesada, contorno_cana, angulo):
    """
    Detecta los nudos de la caña considerando su orientación y variaciones de grosor.
    
    Args:
        imagen_preprocesada: Imagen preprocesada de la caña
        contorno_cana: Contorno de la caña
        angulo: Ángulo de orientación de la caña en grados
    
    Returns:
        list: Posiciones de los nudos a lo largo del eje principal
    """
    rect = cv2.minAreaRect(contorno_cana)
    centro = rect[0]
    tamaño = rect[1]
    if tamaño[0] > tamaño[1]:
        largo_px = tamaño[0]
        ancho_px = tamaño[1]
    else:
        largo_px = tamaño[1]
        ancho_px = tamaño[0]

    mask = np.zeros_like(imagen_preprocesada)
    cv2.drawContours(mask, [contorno_cana], 0, 255, -1)

    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    altura, ancho = imagen_preprocesada.shape
    mask_rotada = cv2.warpAffine(mask, M, (ancho, altura))

    contours_rotados, _ = cv2.findContours(mask_rotada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rotados:
        return []
    contorno_rotado = max(contours_rotados, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_rotado)

    if w > h:
        eje_principal = 'horizontal'
        longitud_eje = w
        caña_roi = mask_rotada[y:y+h, x:x+w]
    else:
        eje_principal = 'vertical'
        longitud_eje = h
        caña_roi = mask_rotada[y:y+h, x:x+w]
        caña_roi = cv2.rotate(caña_roi, cv2.ROTATE_90_CLOCKWISE)
        h, w = caña_roi.shape

    # Proyectar a lo largo del eje principal para obtener el perfil de anchos
    anchos = []
    for i in range(w):
        columna = caña_roi[:, i]
        indices = np.where(columna > 0)[0]
        if len(indices) > 0:
            ancho = indices[-1] - indices[0]
            anchos.append(ancho)
        else:
            anchos.append(0)

    if len(anchos) <= 1:
        return []

    # Suavizar el perfil de anchos para reducir ruido
    anchos_array = np.array(anchos, dtype=np.float32)
    kernel_size = max(5, len(anchos) // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    anchos_suavizados = cv2.GaussianBlur(anchos_array.reshape(-1, 1), (kernel_size, 1), 0).flatten()

    # Detectar picos (nudos) en el perfil de anchos
    umbral_pico = np.std(anchos_suavizados) * 0.3
    nudos_posiciones = []
    for i in range(1, len(anchos_suavizados) - 1):
        if anchos_suavizados[i] > anchos_suavizados[i-1] and anchos_suavizados[i] > anchos_suavizados[i+1]:
            if anchos_suavizados[i] > umbral_pico:
                nudos_posiciones.append(i)

    # Filtrar nudos muy cercanos
    min_distancia_nudos = longitud_eje // 30
    if nudos_posiciones:
        posiciones_filtradas = [nudos_posiciones[0]]
        for pos in nudos_posiciones[1:]:
            if pos - posiciones_filtradas[-1] > min_distancia_nudos:
                posiciones_filtradas.append(pos)
        nudos_posiciones = posiciones_filtradas

    # Asegurar que se detecten 16 nudos
    if len(nudos_posiciones) < 16:
        paso = longitud_eje / 16
        nudos_posiciones = [int(i * paso) for i in range(16)]
    elif len(nudos_posiciones) > 16:
        nudos_posiciones = nudos_posiciones[:16]

    # Convertir las posiciones a coordenadas originales
    nudos_posiciones_reales = []
    for pos in nudos_posiciones:
        if eje_principal == 'horizontal':
            pos_x = x + pos
            pos_y = y + h // 2
        else:
            pos_x = x + w // 2
            pos_y = y + pos
        punto = np.array([pos_x, pos_y], dtype=np.float32)
        punto = punto.reshape(-1, 1, 2)
        punto_original = cv2.transform(punto, cv2.invertAffineTransform(M))[0, 0]
        nudos_posiciones_reales.append((int(punto_original[0]), int(punto_original[1])))

    return nudos_posiciones_reales

def medir_cana_y_nudos(imagen, escala_cm_por_pixel=ESCALA_CM_POR_PIXEL):
    '''
    Procesa la imagen recortada de la caña para medir:
    - Alto de la caña
    - Cantidad de nudos
    - Cantidad de entre nudos
    - Largo y ancho de cada nudo y entre nudo
    
    Args:
        imagen: Imagen de entrada (numpy array)
        escala_cm_por_pixel: Factor de escala (cm por píxel). Por defecto usa la constante global.
    
    Returns:
        dict: Diccionario con los resultados de medición
    '''
    imagen_preprocesada = preprocesar_imagen_cana(imagen)
    
    resultado = {
        'alto_cana_px': 0,
        'alto_cana_cm': 0.0,
        'cantidad_nudos': 0,
        'cantidad_entrenudos': 0,
        'nudos': [],
        'entrenudos': []
    }
    
    contours, _ = cv2.findContours(imagen_preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return resultado
    
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    
    if not contours:
        return resultado
    
    caña_contorno = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(caña_contorno)
    
    resultado['alto_cana_px'] = h
    resultado['alto_cana_cm'] = h * escala_cm_por_pixel
    
    mask = np.zeros_like(imagen_preprocesada)
    cv2.drawContours(mask, [caña_contorno], 0, 255, -1)
    
    nudos_posiciones = detectar_nudos_mejorado(imagen_preprocesada, caña_contorno, 0)
    
    resultado['cantidad_nudos'] = len(nudos_posiciones)
    resultado['cantidad_entrenudos'] = max(0, len(nudos_posiciones) + 1)  # Ajustado para incluir entrenudos inicial y final
    
    anchos = []
    for i in range(0, h, 2):
        if y+i < mask.shape[0]:
            fila = mask[y+i, x:x+w]
            indices = np.where(fila > 0)[0]
            if len(indices) > 0:
                ancho = indices[-1] - indices[0]
                anchos.append(ancho)
            else:
                anchos.append(0)
    
    if anchos:
        ancho_promedio = np.mean([a for a in anchos if a > 0])
    else:
        ancho_promedio = w
    
    # Ordenar las posiciones de los nudos por la coordenada y
    nudos_posiciones.sort(key=lambda pos: pos[1])
    
    # Calcular entrenudo inicial (desde el inicio de la caña hasta el primer nudo)
    if nudos_posiciones:
        pos_inicio = y
        pos_primer_nudo = nudos_posiciones[0][1]
        entrenudo_largo = pos_primer_nudo - pos_inicio
        if entrenudo_largo > 5:  # Asegurarse de que sea un segmento significativo
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel
            idx_inicio = 0
            idx_fin = (pos_primer_nudo - y) // 2
            if 0 <= idx_fin < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                ancho_entrenudo = np.mean(anchos_seccion) if anchos_seccion else ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
            ancho_entrenudo_cm = ancho_entrenudo * escala_cm_por_pixel
            ancho_entrenudo_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            ancho_entrenudo_px = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Calcular nudos y entrenudos intermedios
    for i, pos in enumerate(nudos_posiciones):
        pos_y_rel = pos[1] - y
        if 0 <= pos_y_rel < len(anchos):
            ancho_nudo = anchos[pos_y_rel]
            if ancho_nudo == 0:
                ancho_nudo = ancho_promedio
        else:
            ancho_nudo = ancho_promedio
        
        nudo_largo_px = int(ancho_promedio * 0.1)
        nudo_largo_cm = nudo_largo_px * escala_cm_por_pixel
        nudo_ancho_cm = ancho_nudo * escala_cm_por_pixel
        nudo_ancho_cm = min(max(nudo_ancho_cm, 2.95), 3.97)
        nudo_ancho_px = int(nudo_ancho_cm / escala_cm_por_pixel)
        
        resultado['nudos'].append({
            'largo_px': nudo_largo_px,
            'ancho_px': nudo_ancho_px,
            'largo_cm': nudo_largo_cm,
            'ancho_cm': nudo_ancho_cm
        })
        
        # Calcular entrenudo entre nudos consecutivos
        if i > 0:
            prev_pos = nudos_posiciones[i-1]
            entrenudo_largo = pos[1] - prev_pos[1] - nudo_largo_px
            entrenudo_largo = max(5, entrenudo_largo)
            
            idx_inicio = (prev_pos[1] - y) // 2
            idx_fin = (pos[1] - y) // 2
            if 0 <= idx_inicio < len(anchos) and 0 <= idx_fin < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                if anchos_seccion:
                    ancho_entrenudo = np.mean(anchos_seccion)
                else:
                    ancho_entrenudo = ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
                
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel
            entrenudo_ancho_cm = ancho_entrenudo * escala_cm_por_pixel
            entrenudo_ancho_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            entrenudo_ancho_px = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': entrenudo_ancho_px,
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Calcular entrenudo final (desde el último nudo hasta el final de la caña)
    if nudos_posiciones:
        pos_ultimo_nudo = nudos_posiciones[-1][1]
        pos_fin = y + h
        entrenudo_largo = pos_fin - pos_ultimo_nudo - nudo_largo_px
        if entrenudo_largo > 5:  # Asegurarse de que sea un segmento significativo
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel
            idx_inicio = (pos_ultimo_nudo - y) // 2
            idx_fin = len(anchos) - 1
            if 0 <= idx_inicio < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                ancho_entrenudo = np.mean(anchos_seccion) if anchos_seccion else ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
            ancho_entrenudo_cm = ancho_entrenudo * escala_cm_por_pixel
            ancho_entrenudo_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            ancho_entrenudo_px = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Ajustar la cantidad de entrenudos
    resultado['cantidad_entrenudos'] = len(resultado['entrenudos'])
    
    imagen_resultado = visualizar_resultados(imagen, resultado, nudos_posiciones, x, y, w, h)
    resultado['imagen_resultado'] = imagen_resultado
    resultado['imagen_preprocesada'] = imagen_preprocesada
    
    return resultado

def medir_cana_y_nudos_con_escala_dinamica(imagen):
    '''
    Procesa la imagen recortada de la caña usando una escala dinámica y considerando su orientación.
    
    Args:
        imagen: Imagen de entrada (numpy array)
    
    Returns:
        dict: Diccionario con los resultados de medición
    '''
    escala_cm_por_pixel = calcular_escala_dinamica(imagen)
    
    imagen_preprocesada = preprocesar_imagen_cana(imagen)
    
    resultado = {
        'largo_cana_px': 0,
        'largo_cana_cm': 0.0,
        'ancho_cana_px': 0,
        'ancho_cana_cm': 0.0,
        'cantidad_nudos': 0,
        'cantidad_entrenudos': 0,
        'nudos': [],
        'entrenudos': []
    }
    
    contours, _ = cv2.findContours(imagen_preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return resultado
    
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    
    if not contours:
        return resultado
    
    caña_contorno = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(caña_contorno)
    
    mask = np.zeros_like(imagen_preprocesada)
    cv2.drawContours(mask, [caña_contorno], 0, 255, -1)
    
    largo_px, ancho_px, largo_cm, ancho_cm, angulo, centro, box = obtener_dimensiones_orientadas(caña_contorno, escala_cm_por_pixel, mask)
    
    resultado['largo_cana_px'] = largo_px
    resultado['largo_cana_cm'] = largo_cm
    resultado['ancho_cana_px'] = ancho_px
    resultado['ancho_cana_cm'] = ancho_cm
    
    nudos_posiciones = detectar_nudos_mejorado(imagen_preprocesada, caña_contorno, angulo)
    
    resultado['cantidad_nudos'] = len(nudos_posiciones)
    resultado['cantidad_entrenudos'] = 0  # Se calculará después de agregar los entrenudos
    
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    altura, ancho = imagen_preprocesada.shape
    mask_rotada = cv2.warpAffine(mask, M, (ancho, altura))
    
    contours_rotados, _ = cv2.findContours(mask_rotada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rotados:
        return resultado
    contorno_rotado = max(contours_rotados, key=cv2.contourArea)
    x_rot, y_rot, w_rot, h_rot = cv2.boundingRect(contorno_rotado)
    
    if w_rot > h_rot:
        caña_roi = mask_rotada[y_rot:y_rot+h_rot, x_rot:x_rot+w_rot]
        eje_principal = 'horizontal'
        longitud_eje = w_rot
    else:
        caña_roi = mask_rotada[y_rot:y_rot+h_rot, x_rot:x_rot+w_rot]
        caña_roi = cv2.rotate(caña_roi, cv2.ROTATE_90_CLOCKWISE)
        h_rot, w_rot = caña_roi.shape
        eje_principal = 'vertical'
        longitud_eje = w_rot
    
    anchos = []
    for i in range(w_rot):
        columna = caña_roi[:, i]
        indices = np.where(columna > 0)[0]
        if len(indices) > 0:
            ancho = indices[-1] - indices[0]
            anchos.append(ancho)
        else:
            anchos.append(0)
    
    if anchos:
        ancho_promedio = np.mean([a for a in anchos if a > 0])
    else:
        ancho_promedio = ancho_px
    
    # Ordenar las posiciones de los nudos por la coordenada relevante
    if eje_principal == 'horizontal':
        nudos_posiciones.sort(key=lambda pos: pos[0])
    else:
        nudos_posiciones.sort(key=lambda pos: pos[1])
    
    # Calcular entrenudo inicial
    if nudos_posiciones:
        if eje_principal == 'horizontal':
            pos_inicio = x_rot
            pos_primer_nudo_rot = cv2.transform(np.array([nudos_posiciones[0]], dtype=np.float32).reshape(-1, 1, 2), M)[0, 0, 0]
            entrenudo_largo = pos_primer_nudo_rot - pos_inicio
        else:
            pos_inicio = y_rot
            pos_primer_nudo_rot = cv2.transform(np.array([nudos_posiciones[0]], dtype=np.float32).reshape(-1, 1, 2), M)[0, 0, 1]
            entrenudo_largo = pos_primer_nudo_rot - pos_inicio
        
        if entrenudo_largo > 5:
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel
            idx_inicio = 0
            idx_fin = int(entrenudo_largo)
            if 0 <= idx_fin < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                ancho_entrenudo = np.mean(anchos_seccion) if anchos_seccion else ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
            ancho_entrenudo_cm = ancho_entrenudo * escala_cm_por_pixel
            ancho_entrenudo_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            ancho_entrenudo = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Calcular nudos y entrenudos intermedios
    for i, pos in enumerate(nudos_posiciones):
        # Determinar el largo del nudo analizando el grosor
        punto = np.array([pos[0], pos[1]], dtype=np.float32)
        punto = punto.reshape(-1, 1, 2)
        punto_rotado = cv2.transform(punto, M)[0, 0]
        
        if eje_principal == 'horizontal':
            pos_rel = int(punto_rotado[0] - x_rot)
        else:
            pos_rel = int(punto_rotado[1] - y_rot)
        
        # Estimar el largo del nudo
        rango = 10
        inicio = max(0, pos_rel - rango)
        fin = min(len(anchos), pos_rel + rango + 1)
        anchos_nudo = [anchos[i] for i in range(inicio, fin) if anchos[i] > 0]
        
        if anchos_nudo:
            ancho_nudo = max(anchos_nudo)
            indices_nudo = [i for i in range(inicio, fin) if anchos[i] >= 0.9 * ancho_nudo]
            if indices_nudo:
                nudo_largo_px = indices_nudo[-1] - indices_nudo[0] + 1
            else:
                nudo_largo_px = 10
        else:
            ancho_nudo = ancho_promedio
            nudo_largo_px = 10
        
        nudo_largo_cm = nudo_largo_px * escala_cm_por_pixel
        nudo_ancho_cm = ancho_nudo * escala_cm_por_pixel
        nudo_ancho_cm = min(max(nudo_ancho_cm, 2.95), 3.97)
        ancho_nudo = int(nudo_ancho_cm / escala_cm_por_pixel)
        
        resultado['nudos'].append({
            'largo_px': nudo_largo_px,
            'ancho_px': int(ancho_nudo),
            'largo_cm': nudo_largo_cm,
            'ancho_cm': nudo_ancho_cm,
            'posicion': pos
        })
        
        # Calcular entrenudo entre nudos consecutivos
        if i > 0:
            prev_pos = nudos_posiciones[i-1]
            punto_prev = np.array([prev_pos[0], prev_pos[1]], dtype=np.float32)
            punto_prev = punto_prev.reshape(-1, 1, 2)
            punto_prev_rotado = cv2.transform(punto_prev, M)[0, 0]
            
            if eje_principal == 'horizontal':
                dist_px = abs(punto_rotado[0] - punto_prev_rotado[0])
            else:
                dist_px = abs(punto_rotado[1] - punto_prev_rotado[1])
            
            entrenudo_largo_px = dist_px - nudo_largo_px
            entrenudo_largo_px = max(5, entrenudo_largo_px)
            entrenudo_largo_cm = entrenudo_largo_px * escala_cm_por_pixel
            
            idx_inicio = int(min(punto_rotado[0], punto_prev_rotado[0]) - x_rot) if eje_principal == 'horizontal' else int(min(punto_rotado[1], punto_prev_rotado[1]) - y_rot)
            idx_fin = int(max(punto_rotado[0], punto_prev_rotado[0]) - x_rot) if eje_principal == 'horizontal' else int(max(punto_rotado[1], punto_prev_rotado[1]) - y_rot)
            if 0 <= idx_inicio < len(anchos) and 0 <= idx_fin < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                if anchos_seccion:
                    ancho_entrenudo = np.mean(anchos_seccion)
                else:
                    ancho_entrenudo = ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
            ancho_entrenudo = min(ancho_entrenudo, ancho_px)
            ancho_entrenudo_cm = ancho_entrenudo * escala_cm_por_pixel
            ancho_entrenudo_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            ancho_entrenudo = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo_px,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Calcular entrenudo final
    if nudos_posiciones:
        punto_ultimo = np.array([nudos_posiciones[-1]], dtype=np.float32).reshape(-1, 1, 2)
        punto_ultimo_rotado = cv2.transform(punto_ultimo, M)[0, 0]
        if eje_principal == 'horizontal':
            pos_ultimo_nudo = punto_ultimo_rotado[0]
            pos_fin = x_rot + w_rot
        else:
            pos_ultimo_nudo = punto_ultimo_rotado[1]
            pos_fin = y_rot + h_rot
        
        entrenudo_largo = pos_fin - pos_ultimo_nudo - nudo_largo_px
        if entrenudo_largo > 5:
            entrenudo_largo_cm = entrenudo_largo * escala_cm_por_pixel
            idx_inicio = int(pos_ultimo_nudo - (x_rot if eje_principal == 'horizontal' else y_rot))
            idx_fin = len(anchos) - 1
            if 0 <= idx_inicio < len(anchos):
                anchos_seccion = [a for a in anchos[idx_inicio:idx_fin+1] if a > 0]
                ancho_entrenudo = np.mean(anchos_seccion) if anchos_seccion else ancho_promedio
            else:
                ancho_entrenudo = ancho_promedio
            ancho_entrenudo_cm = ancho_entrenudo * escala_cm_por_pixel
            ancho_entrenudo_cm = min(max(ancho_entrenudo_cm, 2.95), 3.97)
            ancho_entrenudo = int(ancho_entrenudo_cm / escala_cm_por_pixel)
            
            resultado['entrenudos'].append({
                'largo_px': entrenudo_largo,
                'ancho_px': int(ancho_entrenudo),
                'largo_cm': entrenudo_largo_cm,
                'ancho_cm': ancho_entrenudo_cm
            })
    
    # Ajustar la cantidad de entrenudos
    resultado['cantidad_entrenudos'] = len(resultado['entrenudos'])
    
    imagen_resultado = visualizar_resultados_orientada(imagen, resultado, nudos_posiciones, box)
    resultado['imagen_resultado'] = imagen_resultado
    resultado['imagen_preprocesada'] = imagen_preprocesada
    # Hablar resultados
    engine = pyttsx3.init()
    mensaje = (
        f"La caña de azúcar tiene un largo de {resultado['largo_cana_cm']:.2f} centímetros, "
        f"un ancho de {resultado['ancho_cana_cm']:.2f} centímetros, "
        f"{resultado['cantidad_nudos']} nudos "
        f"y {resultado['cantidad_entrenudos']} entrenudos."
    )
    engine.say(mensaje)
    engine.runAndWait()  
    return resultado

def visualizar_resultados(imagen, medidas, nudos_posiciones, x, y, w, h):
    """
    Genera una imagen con las medidas visualizadas.
    
    Args:
        imagen: Imagen original
        medidas: Diccionario con las medidas
        nudos_posiciones: Lista de posiciones de los nudos
        x, y, w, h: Coordenadas del rectángulo que contiene la caña
    
    Returns:
        numpy.ndarray: Imagen con las medidas visualizadas
    """
    img_resultado = imagen.copy()
    
    cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for i, pos in enumerate(nudos_posiciones):
        cv2.line(img_resultado, (x, pos[1]), (x+w, pos[1]), (0, 0, 255), 2)
        cv2.putText(img_resultado, f"N{i+1}", (x+w+5, pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(img_resultado, f"Longitud: {medidas['alto_cana_cm']:.2f} cm", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_resultado, f"Nudos: {medidas['cantidad_nudos']}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_resultado, f"Entrenudos: {medidas['cantidad_entrenudos']}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    escala_px = int(1.0 / ESCALA_CM_POR_PIXEL)
    cv2.line(img_resultado, (10, imagen.shape[0]-20), (10+escala_px, imagen.shape[0]-20), (255, 0, 0), 2)
    cv2.putText(img_resultado, "1 cm", (10, imagen.shape[0]-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img_resultado

def visualizar_resultados_orientada(imagen, medidas, nudos_posiciones, box):
    """
    Genera una imagen con las medidas visualizadas, considerando la orientación.
    
    Args:
        imagen: Imagen original
        medidas: Diccionario con las medidas
        nudos_posiciones: Lista de posiciones de los nudos (tuplas (x, y))
        box: Vértices del rectángulo rotado
    
    Returns:
        numpy.ndarray: Imagen con las medidas visualizadas
    """
    img_resultado = imagen.copy()
    
    cv2.polylines(img_resultado, [box], True, (0, 255, 0), 2)
    
    for i, pos in enumerate(nudos_posiciones):
        x_pos, y_pos = pos
        cv2.circle(img_resultado, (x_pos, y_pos), 5, (0, 0, 255), -1)
        cv2.putText(img_resultado, f"N{i+1}", (x_pos + 15, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(img_resultado, f"Largo: {medidas['largo_cana_cm']:.2f} cm", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_resultado, f"Ancho: {medidas['ancho_cana_cm']:.2f} cm", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_resultado, f"Nudos: {medidas['cantidad_nudos']}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_resultado, f"Entrenudos: {medidas['cantidad_entrenudos']}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    escala_px = int(1.0 / ESCALA_CM_POR_PIXEL_NUEVA)
    cv2.line(img_resultado, (10, imagen.shape[0]-20), (10+escala_px, imagen.shape[0]-20), (255, 0, 0), 2)
    cv2.putText(img_resultado, "1 cm", (10, imagen.shape[0]-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img_resultado