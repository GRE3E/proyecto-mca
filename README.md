# Proyecto MCA - Procesamiento de Imágenes

## Descripción
Este proyecto permite procesar imágenes para extraer regiones de interés, detectar bordes y calcular mediciones basadas en un cuadrado de referencia conocido.

## Requisitos
- Python 3.10
- Git
- OpenCV
- NumPy
- Pillow

## Instalación
1. Abre una terminal o línea de comandos
2. Clona el repositorio usando Git:
   ```bash
   git clone https://github.com/GRE3E/proyecto-mca.git
   cd proyecto-mca
   ```
3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso
Ejecutar el script principal:
```
python src/main.py
```

## Estructura del Proyecto
- `src/`: Código fuente
  - `image_processor.py`: Procesamiento de imágenes
  - `gui.py`: Interfaz gráfica
  - `main.py`: Punto de entrada
  - `utils.py`: Funciones auxiliares
- `data/`: Imágenes
