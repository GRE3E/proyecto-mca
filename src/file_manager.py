import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import re
import numpy as np
import tkinter as tk
from tkinter import ttk, Button

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
        
    def guardar_imagen_raw(self, info_imagen):
        """Guarda una copia de la imagen original en el directorio raw con formato ca_X."""
        # Copiar el archivo
        shutil.copy2(info_imagen['ruta_origen'], info_imagen['destino'])
        
        return info_imagen['nuevo_nombre']
    
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
        
    def guardar_resultados(self, info_resultado):
        """Guarda la imagen procesada en el directorio processed con el formato especificado."""
        # Guardar imagen procesada
        cv2.imwrite(str(info_resultado['ruta_guardado']), info_resultado['imagen'])
        
        return info_resultado['nombre_procesado']
        
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