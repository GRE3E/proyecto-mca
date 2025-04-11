import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import shutil
from PIL import Image

class DetectorBordes:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detector de Bordes")
        self.root.geometry("600x400")
        
        # Encontrar el directorio base del proyecto
        current_dir = Path(__file__).resolve().parent
        while current_dir.name != 'deteccion_cana_azucar' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        if current_dir.name != 'deteccion_cana_azucar':
            raise RuntimeError("No se pudo encontrar el directorio base del proyecto 'deteccion_cana_azucar'")
        
        # Crear directorios si no existen
        self.raw_dir = current_dir / 'data' / 'raw'
        self.processed_dir = current_dir / 'data' / 'processed'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Variables
        self.imagen_path = None
        
        # Crear interfaz
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Título
        titulo = tk.Label(main_frame, text="Detector de Bordes", font=("Arial", 16, "bold"))
        titulo.pack(pady=10)
        
        # Botón para cargar imagen
        btn_cargar = tk.Button(main_frame, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=10)
        
        # Etiqueta para mostrar ruta de la imagen
        self.lbl_ruta = tk.Label(main_frame, text="No se ha seleccionado ninguna imagen", wraplength=500)
        self.lbl_ruta.pack(pady=10)
        
        # Botón para procesar imagen
        self.btn_procesar = tk.Button(main_frame, text="Procesar Imagen", command=self.procesar_imagen, state='disabled')
        self.btn_procesar.pack(pady=10)
        
        # Etiqueta para mostrar estado
        self.lbl_estado = tk.Label(main_frame, text="")
        self.lbl_estado.pack(pady=10)
    
    def cargar_imagen(self):
        try:
            archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
            if archivo:
                self.imagen_path = archivo
                nombre_archivo = os.path.basename(archivo)
                
                # Copiar a data/raw
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nuevo_nombre = f"{timestamp}_{nombre_archivo}"
                destino = self.raw_dir / nuevo_nombre
                shutil.copy2(archivo, destino)
                
                self.lbl_ruta.config(text=f"Imagen cargada: {nombre_archivo}")
                self.btn_procesar.config(state='normal')
                self.lbl_estado.config(text="Imagen guardada en data/raw")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")
    
    def ordenar_puntos(self, puntos):
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
        y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
        x1_order = y_order[:2]
        x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
        x2_order = y_order[2:4]
        x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
        return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]
    
    def roi(self, image, ancho_max=1920, alto_max=1080):
        imagen_alineada = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        
        for c in cnts:
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                puntos = self.ordenar_puntos(approx)
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
    
    def procesar_imagen(self):
        try:
            if not self.imagen_path:
                raise ValueError("No se ha seleccionado ninguna imagen")
            
            # Leer imagen con OpenCV y PIL
            imagen = cv2.imread(self.imagen_path)
            if imagen is None:
                raise ValueError("No se pudo leer la imagen")
            
            # Reducir resolución para procesamiento
            max_dimension = 800
            height, width = imagen.shape[:2]
            scale = min(max_dimension/width, max_dimension/height)
            if scale < 1:
                new_width = int(width * scale)
                new_height = int(height * scale)
                imagen = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Procesar imagen manteniendo la relación de aspecto
            imagen_A4 = self.roi(imagen)
            if imagen_A4 is None:
                imagen_A4 = imagen.copy()
            
            # Procesar imagen para detección de bordes usando NumPy para mejor rendimiento
            # Convertir la imagen recortada a array de NumPy
            img_array = np.array(Image.fromarray(cv2.cvtColor(imagen_A4, cv2.COLOR_BGR2RGB)))
            
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
            
            # Convertir array procesado a imagen PIL y luego a OpenCV para procesamiento
            pil_image = Image.fromarray(output_array.astype('uint8'))
            imagen_bordes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Liberar memoria
            del img_array
            del output_array
            del luminosity
            del diff_x
            del diff_y
            
            # Detectar contornos en la imagen con bordes
            imagen_bordes_gray = cv2.cvtColor(imagen_bordes, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imagen_bordes_gray, 127, 255, cv2.THRESH_BINARY)
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Encontrar el cuadrado de referencia (asumimos que es el contorno más grande)
            if contornos:
                # Ordenar contornos por área
                contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
                
                # El primer contorno debería ser el cuadrado de referencia
                cnt_referencia = contornos[0]
                x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(cnt_referencia)
                
                # Calcular la relación metros/píxeles usando el cuadrado de 1.64m
                metros_por_pixel = 1.64 / max(w_ref, h_ref)
                
                # Dibujar y medir cada contorno
                for cnt in contornos:
                    # Obtener el rectángulo delimitador
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Calcular dimensiones en metros
                    ancho_m = w * metros_por_pixel
                    alto_m = h * metros_por_pixel
                    
                    # Dibujar rectángulo
                    cv2.rectangle(imagen_bordes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Dibujar líneas de medición
                    cv2.line(imagen_bordes, (x, y + h//2), (x + w, y + h//2), (255, 0, 0), 1)
                    cv2.line(imagen_bordes, (x + w//2, y), (x + w//2, y + h), (255, 0, 0), 1)
                    
                    # Mostrar medidas
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(imagen_bordes, f'{ancho_m:.2f}m', (x, y-5), font, 0.5, (255, 255, 255), 1)
                    cv2.putText(imagen_bordes, f'{alto_m:.2f}m', (x+w+5, y+h//2), font, 0.5, (255, 255, 255), 1)
                    
                # Marcar el cuadrado de referencia
                cv2.rectangle(imagen_bordes, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (0, 0, 255), 2)
                cv2.putText(imagen_bordes, 'Referencia (1.64m)', (x_ref, y_ref-10), font, 0.5, (0, 0, 255), 1)
            
            # Crear imagen en blanco para las mediciones
            imagen_mediciones = np.zeros_like(imagen_A4)
            
            # Dibujar las mediciones en la imagen en blanco
            font = cv2.FONT_HERSHEY_SIMPLEX
            for cnt in contornos:
                x, y, w, h = cv2.boundingRect(cnt)
                ancho_m = w * metros_por_pixel
                alto_m = h * metros_por_pixel
                
                # Dibujar rectángulo y líneas en blanco
                cv2.rectangle(imagen_mediciones, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.line(imagen_mediciones, (x, y + h//2), (x + w, y + h//2), (255, 255, 255), 1)
                cv2.line(imagen_mediciones, (x + w//2, y), (x + w//2, y + h), (255, 255, 255), 1)
                
                # Agregar texto de mediciones en blanco
                cv2.putText(imagen_mediciones, f'{ancho_m:.2f}m', (x, y-5), font, 0.5, (255, 255, 255), 1)
                cv2.putText(imagen_mediciones, f'{alto_m:.2f}m', (x+w+5, y+h//2), font, 0.5, (255, 255, 255), 1)
            
            # Marcar el cuadrado de referencia en la imagen de mediciones
            cv2.rectangle(imagen_mediciones, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (255, 255, 255), 2)
            cv2.putText(imagen_mediciones, 'Referencia (1.64m)', (x_ref, y_ref-10), font, 0.5, (255, 255, 255), 1)
            
            # Guardar imágenes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = os.path.basename(self.imagen_path)
            
            # Guardar imagen procesada original
            nombre_procesado = f"processed_{timestamp}_{nombre_archivo}"
            ruta_guardado = self.processed_dir / nombre_procesado
            cv2.imwrite(str(ruta_guardado), imagen_A4)
            
            # Guardar imagen con bordes
            nombre_bordes = f"processed_bordes_{timestamp}_{nombre_archivo}"
            ruta_bordes = self.processed_dir / nombre_bordes
            pil_image.save(str(ruta_bordes))
            
            # Guardar imagen de mediciones en processed
            nombre_mediciones = f"mediciones_{timestamp}_{nombre_archivo}"
            ruta_mediciones = self.processed_dir / nombre_mediciones
            cv2.imwrite(str(ruta_mediciones), imagen_mediciones)
            
            self.lbl_estado.config(text=f"Imágenes guardadas como:\n{nombre_procesado}\n{nombre_bordes}\n{nombre_mediciones}")
            messagebox.showinfo("Éxito", "Imágenes procesadas y guardadas correctamente")
            #else:
             #   raise ValueError("No se detectó ningún objeto verde")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
    
    def iniciar(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DetectorBordes()
    app.iniciar()