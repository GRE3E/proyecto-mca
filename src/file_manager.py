import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import re

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
    
    def guardar_imagen_raw(self, ruta_origen):
        """Guarda una copia de la imagen original en el directorio raw con formato ca_X."""
        # Obtener extensión del archivo original
        _, extension = os.path.splitext(ruta_origen)
        
        # Obtener el siguiente número secuencial
        siguiente_numero = self._obtener_siguiente_numero(self.raw_dir, r'ca_([0-9]+)')
        
        # Crear el nuevo nombre
        nuevo_nombre = f"ca_{siguiente_numero}{extension}"
        destino = self.raw_dir / nuevo_nombre
        
        # Copiar el archivo
        shutil.copy2(ruta_origen, destino)
        
        return nuevo_nombre
    
    def guardar_resultados(self, imagen_A4, imagen_bordes, imagen_mediciones, ruta_origen):
        """Guarda las imágenes procesadas en el directorio processed con los formatos especificados."""
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
        
        # Guardar imagen procesada original (reducción)
        nombre_procesado = f"reduccion_bordes_ca_{numero}{extension}"
        ruta_guardado = self.processed_dir / nombre_procesado
        cv2.imwrite(str(ruta_guardado), imagen_A4)
        
        # Guardar imagen con bordes (monocromático)
        nombre_bordes = f"monocromatico_bordes_ca_{numero}{extension}"
        ruta_bordes = self.processed_dir / nombre_bordes
        if isinstance(imagen_bordes, Image.Image):
            imagen_bordes.save(str(ruta_bordes))
        else:
            cv2.imwrite(str(ruta_bordes), imagen_bordes)
        
        # Guardar imagen de mediciones
        nombre_mediciones = f"mediccion_bordes_filtro_ca_{numero}{extension}"
        ruta_mediciones = self.processed_dir / nombre_mediciones
        cv2.imwrite(str(ruta_mediciones), imagen_mediciones)
        
        return [nombre_procesado, nombre_bordes, nombre_mediciones]