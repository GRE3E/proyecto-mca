import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image

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
    
    def guardar_imagen_raw(self, ruta_origen):
        """Guarda una copia de la imagen original en el directorio raw con formato ca_(numero)."""
        nombre_archivo = os.path.basename(ruta_origen)
        # Extraer número de archivo o generar uno basado en timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Usar los últimos 6 dígitos del timestamp como número
        numero = timestamp[-6:]
        nuevo_nombre = f"ca_{numero}.jpg"
        destino = self.raw_dir / nuevo_nombre
        shutil.copy2(ruta_origen, destino)
        return nuevo_nombre
    
    def guardar_resultados(self, imagen_A4, imagen_bordes, imagen_mediciones, ruta_origen):
        """Guarda las imágenes procesadas en el directorio processed con los formatos solicitados."""
        # Extraer número de archivo o generar uno basado en timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Usar los últimos 6 dígitos del timestamp como número
        numero = timestamp[-6:]
        
        # Guardar imagen procesada original (reducción)
        nombre_procesado = f"reduccion_bordes_ca{numero}.jpg"
        ruta_guardado = self.processed_dir / nombre_procesado
        cv2.imwrite(str(ruta_guardado), imagen_A4)
        
        # Guardar imagen con bordes (monocromático)
        nombre_bordes = f"monocromatico_bordes_ca{numero}.jpg"
        ruta_bordes = self.processed_dir / nombre_bordes
        if isinstance(imagen_bordes, Image.Image):
            imagen_bordes.save(str(ruta_bordes))
        else:
            cv2.imwrite(str(ruta_bordes), imagen_bordes)
        
        # Guardar imagen de mediciones
        nombre_mediciones = f"mediccion_bordes_filtro_ca{numero}.jpg"
        ruta_mediciones = self.processed_dir / nombre_mediciones
        cv2.imwrite(str(ruta_mediciones), imagen_mediciones)
        
        return [nombre_procesado, nombre_bordes, nombre_mediciones]