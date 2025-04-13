import cv2
import numpy as np

def calcular_mediciones(imagen_bordes, imagen_original):
    imagen_bordes_cv = cv2.cvtColor(np.array(imagen_bordes), cv2.COLOR_RGB2BGR)
    imagen_bordes_gray = cv2.cvtColor(imagen_bordes_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imagen_bordes_gray, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagen_mediciones = np.zeros_like(imagen_original)
    metros_por_pixel = 0

    if contornos:
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        cnt_referencia = contornos[0]
        x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(cnt_referencia)
        metros_por_pixel = 1.64 / max(w_ref, h_ref)

        for cnt in contornos:
            dibujar_mediciones(imagen_bordes_cv, cnt, metros_por_pixel)
        
        cv2.rectangle(imagen_bordes_cv, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (0, 0, 255), 2)
        cv2.putText(imagen_bordes_cv, 'Referencia (1.64m)', (x_ref, y_ref-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for cnt in contornos:
            dibujar_mediciones(imagen_mediciones, cnt, metros_por_pixel, color=(255, 255, 255))
        
        cv2.rectangle(imagen_mediciones, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (255, 255, 255), 2)
        cv2.putText(imagen_mediciones, 'Referencia (1.64m)', (x_ref, y_ref-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return imagen_bordes_cv, imagen_mediciones

def dibujar_mediciones(imagen, contorno, metros_por_pixel, color=(0, 255, 0)):
    x, y, w, h = cv2.boundingRect(contorno)
    ancho_m = w * metros_por_pixel
    alto_m = h * metros_por_pixel
    cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 2)
    cv2.line(imagen, (x, y + h//2), (x + w, y + h//2), (255, 0, 0) if color != (255, 255, 255) else color, 1)
    cv2.line(imagen, (x + w//2, y), (x + w//2, y + h), (255, 0, 0) if color != (255, 255, 255) else color, 1)
    cv2.putText(imagen, f'{ancho_m:.2f}m', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(imagen, f'{alto_m:.2f}m', (x+w+5, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
