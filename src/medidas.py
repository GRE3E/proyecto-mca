import cv2
import numpy as np

def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def roi(image, ancho, alto):
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
            pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_alineada = cv2.warpPerspective(image, M, (ancho, alto))
    return imagen_alineada

# Cargar imagen proporcionada
imagen = cv2.imread('src/prueba2.png')

if imagen is not None:
    imagen_A4 = roi(imagen, ancho=1080, alto=720)
    if imagen_A4 is None:
        imagen_A4 = imagen.copy()

    imagenHSV = cv2.cvtColor(imagen_A4, cv2.COLOR_BGR2HSV)

    # Rango de verde
    verdeBajo = np.array([36, 25, 25], np.uint8)
    verdeAlto = np.array([86, 255, 255], np.uint8)

    maskVerde = cv2.inRange(imagenHSV, verdeBajo, verdeAlto)
    cnts = cv2.findContours(maskVerde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if cnts:
        c = cnts[0]
        x, y, w, h = cv2.boundingRect(c)

        # Conversiones a centímetros
        cm_por_pixel_ancho = 21.0 / imagen_A4.shape[1]
        cm_por_pixel_alto = 29.7 / imagen_A4.shape[0]

        ancho_cm = w * cm_por_pixel_ancho
        alto_cm = h * cm_por_pixel_alto

        # Dibujar rectángulo
        cv2.rectangle(imagen_A4, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Líneas horizontales y verticales con color claro
        color_linea = (255, 255, 255)  # blanco
        centro_izq = (x, y + h // 2)
        centro_der = (x + w, y + h // 2)
        centro_arriba = (x + w // 2, y)
        centro_abajo = (x + w // 2, y + h)

        cv2.line(imagen_A4, centro_izq, centro_der, color_linea, 2)
        cv2.line(imagen_A4, centro_arriba, centro_abajo, color_linea, 2)

        # Texto de dimensiones externas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color_fuera = (0, 0, 255)  # rojo

        cv2.putText(imagen_A4, f"Ancho: {ancho_cm:.2f} cm", (x + w + 10, y + 30), font, font_scale, color_fuera, thickness)
        cv2.putText(imagen_A4, f"Alto: {alto_cm:.2f} cm", (x + w + 10, y + 60), font, font_scale, color_fuera, thickness)

        # Texto sobre las líneas internas en blanco con sombra negra
        color_texto_dentro = (255, 255, 255)  # blanco
        sombra = (0, 0, 0)  # negro

        mid_h = ((centro_izq[0] + centro_der[0]) // 2, centro_izq[1] - 10)
        mid_v = (centro_arriba[0] + 10, (centro_arriba[1] + centro_abajo[1]) // 2)

        # Posiciones para los textos de medidas internas (más separados)
        offset_texto = 40  # distancia extra para separar

        mid_h = (
            (centro_izq[0] + centro_der[0]) // 2 - 50,
            centro_izq[1] - offset_texto
        )
        mid_v = (
            centro_arriba[0] + offset_texto,
            (centro_arriba[1] + centro_abajo[1]) // 2 + 30
        )

        # Sombra (negro) para mejor visibilidad
        cv2.putText(imagen_A4, f"{ancho_cm:.2f} cm", (mid_h[0] + 1, mid_h[1] + 1), font, 0.6, sombra, 2)
        cv2.putText(imagen_A4, f"{alto_cm:.2f} cm", (mid_v[0] + 1, mid_v[1] + 1), font, 0.6, sombra, 2)

        # Texto principal (blanco)
        cv2.putText(imagen_A4, f"{ancho_cm:.2f} cm", mid_h, font, 0.6, color_texto_dentro, 1)
        cv2.putText(imagen_A4, f"{alto_cm:.2f} cm", mid_v, font, 0.6, color_texto_dentro, 1)

        # Mostrar imagen
        cv2.imshow("Distancias Claras", imagen_A4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se detectó ningún objeto verde.")
else:
    print("No se pudo cargar la imagen.")
