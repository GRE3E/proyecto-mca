import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import re
import numpy as np
import tkinter as tk
from tkinter import ttk, Button, filedialog
from typing import List, Dict, Any, Optional, Tuple
import threading

class FileManager:
    """Clase para manejar operaciones de archivos y directorios."""
    
    def __init__(self):
        # Encontrar el directorio base del proyecto
        current_dir = Path(__file__).resolve().parent
        while current_dir.name != 'proyecto-mca' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        if current_dir.name != 'proyecto-mca':
            raise RuntimeError("No se pudo encontrar el directorio base del proyecto 'proyecto-mca'")
        
        # Crear directorios si no existen
        self.raw_dir = current_dir / 'data' / 'raw'
        self.processed_dir = current_dir / 'data' / 'processed'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Extensiones de imagen válidas
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    def _obtener_siguiente_numero(self, directorio, patron):
        """Obtiene el siguiente número secuencial para nombrar archivos."""
        archivos = list(directorio.glob('*'))
        numeros = []
        
        for archivo in archivos:
            match = re.search(patron, archivo.name)
            if match and match.group(1).isdigit():
                numeros.append(int(match.group(1)))
        
        return max(numeros, default=0) + 1
    
    def preparar_imagen_raw(self, ruta_origen):
        """Prepara la información para guardar una imagen en raw, pero no la guarda aún."""
        # Obtener extensión del archivo original
        _, extension = os.path.splitext(ruta_origen)
        
        # Obtener el siguiente número secuencial
        siguiente_numero = self._obtener_siguiente_numero(self.raw_dir, r'ca_([0-9]+)')
        
        # Crear el nuevo nombre
        nuevo_nombre = f"ca_{siguiente_numero}{extension}"
        destino = self.raw_dir / nuevo_nombre
        
        return {
            'ruta_origen': ruta_origen,
            'destino': destino,
            'nuevo_nombre': nuevo_nombre
        }
    
    def preparar_multiples_imagenes_raw(self, rutas_origen: List[str]) -> List[Dict[str, Any]]:
        """Prepara información para guardar múltiples imágenes en raw."""
        resultados = []
        
        # Obtener el número inicial
        siguiente_numero = self._obtener_siguiente_numero(self.raw_dir, r'ca_([0-9]+)')
        
        for i, ruta in enumerate(rutas_origen):
            _, extension = os.path.splitext(ruta)
            nuevo_nombre = f"ca_{siguiente_numero + i}{extension}"
            destino = self.raw_dir / nuevo_nombre
            
            resultados.append({
                'ruta_origen': ruta,
                'destino': destino,
                'nuevo_nombre': nuevo_nombre
            })
        
        return resultados
        
    def guardar_imagen_raw(self, info_imagen):
        """Guarda una copia de la imagen original en el directorio raw con formato ca_X."""
        # Copiar el archivo
        shutil.copy2(info_imagen['ruta_origen'], info_imagen['destino'])
        
        return info_imagen['nuevo_nombre']
    
    def guardar_multiples_imagenes_raw(self, info_imagenes: List[Dict[str, Any]]) -> List[str]:
        """Guarda múltiples imágenes en el directorio raw."""
        nombres_guardados = []
        
        for info in info_imagenes:
            try:
                shutil.copy2(info['ruta_origen'], info['destino'])
                nombres_guardados.append(info['nuevo_nombre'])
            except Exception as e:
                print(f"Error al guardar {info['ruta_origen']}: {e}")
        
        return nombres_guardados
    
    def preparar_resultados(self, imagen_A4, ruta_origen):
        """Prepara la información para guardar la imagen procesada, pero no la guarda aún."""
        # Extraer el número del archivo original si es un archivo ca_X
        nombre_archivo = os.path.basename(ruta_origen)
        match = re.search(r'ca_([0-9]+)', nombre_archivo)
        
        if match:
            numero = match.group(1)
        else:
            # Si no es un archivo ca_X, obtener el siguiente número disponible
            numero = str(self._obtener_siguiente_numero(self.processed_dir, r'reduccion_bordes_ca_([0-9]+)'))
        
        # Obtener extensión del archivo original
        _, extension = os.path.splitext(nombre_archivo)
        
        # Preparar nombre y ruta para guardar
        nombre_procesado = f"reduccion_bordes_ca_{numero}{extension}"
        ruta_guardado = self.processed_dir / nombre_procesado
        
        return {
            'imagen': imagen_A4,
            'ruta_guardado': ruta_guardado,
            'nombre_procesado': nombre_procesado
        }
    
    def preparar_multiples_resultados(self, imagenes_procesadas: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """Prepara información para guardar múltiples imágenes procesadas."""
        resultados = []
        
        for imagen, ruta_origen in imagenes_procesadas:
            info = self.preparar_resultados(imagen, ruta_origen)
            resultados.append(info)
        
        return resultados
        
    def guardar_resultados(self, info_resultado):
        """Guarda la imagen procesada en el directorio processed con el formato especificado."""
        # Guardar imagen procesada
        cv2.imwrite(str(info_resultado['ruta_guardado']), info_resultado['imagen'])
        
        return info_resultado['nombre_procesado']
    
    def guardar_multiples_resultados(self, info_resultados: List[Dict[str, Any]]) -> List[str]:
        """Guarda múltiples imágenes procesadas."""
        nombres_guardados = []
        
        for info in info_resultados:
            try:
                cv2.imwrite(str(info['ruta_guardado']), info['imagen'])
                nombres_guardados.append(info['nombre_procesado'])
            except Exception as e:
                print(f"Error al guardar resultado {info['ruta_guardado']}: {e}")
        
        return nombres_guardados
        
    def mostrar_vista_previa(self, imagen, info_imagen_raw=None, info_resultado=None, callback_guardar=None, callback_cancelar=None):
        """Muestra una vista previa de la imagen procesada con opciones para guardar o cancelar."""
        # Crear ventana de vista previa
        ventana = tk.Toplevel()
        ventana.title("Vista Previa - Confirmar Guardado")
        ventana.geometry("800x600")
        ventana.configure(bg="#F1F6F9")
        
        # Convertir imagen de OpenCV a formato para tkinter
        if isinstance(imagen, np.ndarray):
            # Convertir de BGR a RGB
            if len(imagen.shape) == 3 and imagen.shape[2] == 3:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen_rgb = imagen
                
            # Redimensionar si es muy grande
            alto, ancho = imagen_rgb.shape[:2]
            max_size = 700
            if alto > max_size or ancho > max_size:
                if alto > ancho:
                    nuevo_alto = max_size
                    nuevo_ancho = int(ancho * (max_size / alto))
                else:
                    nuevo_ancho = max_size
                    nuevo_alto = int(alto * (max_size / ancho))
                imagen_rgb = cv2.resize(imagen_rgb, (nuevo_ancho, nuevo_alto))
            
            # Convertir a formato PIL y luego a PhotoImage
            imagen_pil = Image.fromarray(imagen_rgb)
            from PIL import ImageTk
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
        else:
            # Si ya es un objeto PIL Image
            imagen_tk = ImageTk.PhotoImage(imagen)
        
        # Mostrar imagen
        lbl_imagen = ttk.Label(ventana, image=imagen_tk)
        lbl_imagen.image = imagen_tk  # Mantener referencia
        lbl_imagen.pack(pady=20)
        
        # Frame para botones
        frame_botones = ttk.Frame(ventana, padding=10)
        frame_botones.pack(pady=10)
        
        # Función para guardar
        def guardar():
            if callback_guardar:
                callback_guardar(info_imagen_raw, info_resultado)
            ventana.destroy()
        
        # Función para cancelar
        def cancelar():
            if callback_cancelar:
                callback_cancelar()
            ventana.destroy()
        
        # Botones
        btn_guardar = ttk.Button(frame_botones, text="✅ Guardar", command=guardar)
        btn_guardar.pack(side=tk.LEFT, padx=10)
        
        btn_cancelar = ttk.Button(frame_botones, text="❌ Cancelar", command=cancelar)
        btn_cancelar.pack(side=tk.LEFT, padx=10)
        
        # Centrar ventana
        ventana.update_idletasks()
        ancho_ventana = ventana.winfo_width()
        alto_ventana = ventana.winfo_height()
        x = (ventana.winfo_screenwidth() // 2) - (ancho_ventana // 2)
        y = (ventana.winfo_screenheight() // 2) - (alto_ventana // 2)
        ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
        
        # Hacer modal
        ventana.transient(ventana.master)
        ventana.grab_set()
        ventana.wait_window()
    
    def mostrar_resumen_procesamiento(self, total_archivos: int, procesados: int, errores: int = 0):
        """Muestra un resumen del procesamiento por lotes."""
        ventana = tk.Toplevel()
        ventana.title("Resumen de Procesamiento")
        ventana.geometry("400x300")
        ventana.configure(bg="#F1F6F9")
        
        # Frame principal
        frame = ttk.Frame(ventana, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Resumen de Procesamiento", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Información
        ttk.Label(frame, text=f"Total de archivos: {total_archivos}", font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        ttk.Label(frame, text=f"Archivos procesados: {procesados}", font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        
        if errores > 0:
            ttk.Label(frame, text=f"Errores: {errores}", font=("Arial", 12, "bold"), foreground="red").pack(anchor=tk.W, pady=5)
        else:
            ttk.Label(frame, text="Errores: 0", font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        
        # Mensaje de éxito
        if errores == 0 and procesados > 0:
            ttk.Label(frame, text="¡Procesamiento completado con éxito!", 
                      font=("Arial", 12, "bold"), foreground="green").pack(pady=15)
        elif procesados > 0:
            ttk.Label(frame, text="Procesamiento completado con algunos errores", 
                      font=("Arial", 12, "bold"), foreground="orange").pack(pady=15)
        else:
            ttk.Label(frame, text="No se pudo procesar ningún archivo", 
                      font=("Arial", 12, "bold"), foreground="red").pack(pady=15)
        
        # Botón de cerrar
        ttk.Button(frame, text="Cerrar", command=ventana.destroy).pack(pady=10)
        
        # Centrar ventana
        ventana.update_idletasks()
        ancho_ventana = ventana.winfo_width()
        alto_ventana = ventana.winfo_height()
        x = (ventana.winfo_screenwidth() // 2) - (ancho_ventana // 2)
        y = (ventana.winfo_screenheight() // 2) - (alto_ventana // 2)
        ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
        
        # Hacer modal
        ventana.transient(ventana.master)
        ventana.grab_set()
        ventana.wait_window()
    
    def seleccionar_carpeta(self) -> Optional[str]:
        """Abre un diálogo para seleccionar una carpeta."""
        ruta_carpeta = filedialog.askdirectory(title="Seleccionar carpeta con imágenes")
        return ruta_carpeta if ruta_carpeta else None
    
    def obtener_imagenes_de_carpeta(self, ruta_carpeta: str) -> List[str]:
        """Obtiene todas las rutas de imágenes dentro de una carpeta."""
        if not ruta_carpeta or not os.path.isdir(ruta_carpeta):
            return []
        
        rutas_imagenes = []
        
        for root, _, files in os.walk(ruta_carpeta):
            for file in files:
                extension = os.path.splitext(file)[1].lower()
                if extension in self.valid_extensions:
                    rutas_imagenes.append(os.path.join(root, file))
        
        return rutas_imagenes
    
    def procesar_carpeta(self, ruta_carpeta: str, funcion_procesar, mostrar_progreso=True):
        """
        Procesa todas las imágenes en una carpeta usando la función de procesamiento proporcionada.
        
        Args:
            ruta_carpeta: Ruta a la carpeta con imágenes
            funcion_procesar: Función que recibe la ruta de una imagen y la procesa
            mostrar_progreso: Si es True, muestra una ventana de progreso
        """
        # Obtener todas las imágenes en la carpeta
        rutas_imagenes = self.obtener_imagenes_de_carpeta(ruta_carpeta)
        
        if not rutas_imagenes:
            tk.messagebox.showinfo("Información", "No se encontraron imágenes en la carpeta seleccionada.")
            return
        
        total_imagenes = len(rutas_imagenes)
        
        # Si hay que mostrar progreso, crear ventana
        if mostrar_progreso:
            ventana_progreso = tk.Toplevel()
            ventana_progreso.title("Procesando imágenes")
            ventana_progreso.geometry("400x150")
            
            ttk.Label(ventana_progreso, text="Procesando imágenes...").pack(pady=10)
            
            progreso = ttk.Progressbar(ventana_progreso, orient=tk.HORIZONTAL, 
                                     length=300, mode='determinate')
            progreso.pack(pady=10)
            progreso['maximum'] = total_imagenes
            
            lbl_estado = ttk.Label(ventana_progreso, text=f"0/{total_imagenes}")
            lbl_estado.pack(pady=5)
            
            # Botón para cancelar
            btn_cancelar = ttk.Button(ventana_progreso, text="Cancelar", command=ventana_progreso.destroy)
            btn_cancelar.pack(pady=10)
            
            # Variable para seguir progreso
            progreso_actual = {'completados': 0, 'errores': 0, 'cancelado': False}
            
            # Función para actualizar progreso
            def actualizar_progreso(completados, errores=0):
                if ventana_progreso.winfo_exists():
                    progreso_actual['completados'] = completados
                    progreso_actual['errores'] = errores
                    progreso['value'] = completados
                    lbl_estado.config(text=f"{completados}/{total_imagenes}")
                    ventana_progreso.update()
            
            # Función para procesar en segundo plano
            def procesar_en_segundo_plano():
                resultados_procesados = []
                errores = 0
                
                # Información para las imágenes raw
                info_imagenes_raw = self.preparar_multiples_imagenes_raw(rutas_imagenes)
                
                # Guardar las imágenes raw
                self.guardar_multiples_imagenes_raw(info_imagenes_raw)
                
                # Procesar cada imagen
                for i, (ruta, info_raw) in enumerate(zip(rutas_imagenes, info_imagenes_raw)):
                    if ventana_progreso.winfo_exists():
                        try:
                            # Procesar la imagen
                            imagen_procesada = funcion_procesar(ruta)
                            
                            # Preparar para guardar el resultado
                            info_resultado = self.preparar_resultados(imagen_procesada, info_raw['destino'])
                            
                            # Guardar el resultado
                            self.guardar_resultados(info_resultado)
                            
                            # Actualizar progreso
                            actualizar_progreso(i + 1, errores)
                        except Exception as e:
                            print(f"Error al procesar {ruta}: {e}")
                            errores += 1
                            actualizar_progreso(i + 1, errores)
                    else:
                        # Si la ventana se cerró, cancelar el procesamiento
                        progreso_actual['cancelado'] = True
                        break
                
                # Cerrar ventana de progreso si sigue abierta
                if ventana_progreso.winfo_exists():
                    ventana_progreso.destroy()
                
                # Mostrar resumen si no se canceló
                if not progreso_actual['cancelado']:
                    self.mostrar_resumen_procesamiento(
                        total_imagenes, 
                        progreso_actual['completados'], 
                        progreso_actual['errores']
                    )
            
            # Iniciar procesamiento en segundo plano
            threading.Thread(target=procesar_en_segundo_plano, daemon=True).start()
            
            # Hacer modal
            ventana_progreso.transient(ventana_progreso.master)
            ventana_progreso.grab_set()
            ventana_progreso.wait_window()
        else:
            # Procesamiento sin interfaz de progreso
            resultados_procesados = []
            errores = 0
            
            # Información para las imágenes raw
            info_imagenes_raw = self.preparar_multiples_imagenes_raw(rutas_imagenes)
            
            # Guardar las imágenes raw
            self.guardar_multiples_imagenes_raw(info_imagenes_raw)
            
            # Procesar cada imagen
            for i, (ruta, info_raw) in enumerate(zip(rutas_imagenes, info_imagenes_raw)):
                try:
                    # Procesar la imagen
                    imagen_procesada = funcion_procesar(ruta)
                    
                    # Preparar para guardar el resultado
                    info_resultado = self.preparar_resultados(imagen_procesada, info_raw['destino'])
                    
                    # Guardar el resultado
                    self.guardar_resultados(info_resultado)
                except Exception as e:
                    print(f"Error al procesar {ruta}: {e}")
                    errores += 1
            
            # Mostrar resumen
            self.mostrar_resumen_procesamiento(total_imagenes, total_imagenes - errores, errores)