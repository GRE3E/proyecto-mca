# Proyecto MCA - Sistema de Clasificación de Caña de Azúcar

## Descripción
Este proyecto implementa un sistema de clasificación de imágenes para identificar caña de azúcar, incluyendo procesamiento de imágenes para extraer regiones de interés, detección de bordes y clasificación mediante aprendizaje profundo.

## Requisitos
- Python 3.10
- Git
- OpenCV
- NumPy
- Pillow
- TensorFlow

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/GRE3E/proyecto-mca.git
   cd proyecto-mca
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Estructura del Proyecto

```
proyecto-mca/
├── README.md         # Documentación del proyecto
├── requirements.txt  # Dependencias del proyecto
├── .gitignore        # Archivos y directorios a ignorar por Git
├── checkpoints/      # Modelos entrenados guardados
├── data/
│   ├── model_training/ # Datos para entrenamiento y validación del modelo
│   │   ├── train/     # Conjunto de entrenamiento
│   │   │   ├── ca/    # Imágenes de caña
│   │   │   └── no_ca/ # Imágenes que no son caña
│   │   └── val/       # Conjunto de validación
│   │       ├── ca/  
│   │       └── no_ca/
│   ├── processed/   # Imágenes después de aplicar procesamiento
│   └── raw/        # Imágenes originales sin procesar
└── src/
    ├── app_main.py  # Menú principal para interactuar con el sistema
    ├── img_proc/    # Módulos para procesamiento de imágenes
    │   ├── edge_detection.py # Detección de bordes
    │   ├── esc_grises.py     # Conversión a escala de grises
    │   ├── main_processor.py # Lógica principal de procesamiento de imágenes
    │   └── roi_extraction.py # Extracción de regiones de interés
    ├── model/       # Módulos relacionados con el modelo de aprendizaje profundo
    │   ├── inference.py    # Lógica para realizar inferencias con el modelo entrenado
    │   └── training.py     # Lógica para entrenar el modelo
    ├── gui/         # Módulos para la interfaz gráfica de usuario
    │   └── app.py        # Implementación de la interfaz gráfica
    ├── file_manager.py # Gestión de archivos y directorios
    ├── main.py      # Punto de entrada principal del flujo de trabajo
    └── utils.py     # Funciones de utilidad
```

## Funcionamiento del Proyecto

El sistema de clasificación de caña de azúcar sigue un flujo de trabajo general que incluye los siguientes pasos:

1.  **Carga de Imágenes:** Las imágenes sin procesar se cargan desde el directorio `data/raw/`.
2.  **Procesamiento de Imágenes:** Las imágenes pasan por un pipeline de procesamiento en el módulo `img_proc/`. Esto puede incluir conversión a escala de grises, detección de bordes y extracción de regiones de interés (ROI).
3.  **Preparación de Datos:** Para el entrenamiento, las imágenes procesadas se organizan en `data/model_training/` en conjuntos de entrenamiento y validación, separadas por clases (`ca` y `no_ca`).
4.  **Entrenamiento del Modelo:** El módulo `model/training.py` se encarga de entrenar un modelo de aprendizaje profundo utilizando los datos preparados. Los modelos entrenados se guardan en `checkpoints/`.
5.  **Inferencia/Clasificación:** El módulo `model/inference.py` utiliza un modelo entrenado para clasificar nuevas imágenes procesadas.
6.  **Gestión de Archivos:** El módulo `file_manager.py` ayuda en la organización y manejo de los archivos del proyecto.
7.  **Interfaz de Usuario:** La interfaz gráfica en `gui/app.py` permite interactuar con el sistema, posiblemente para cargar imágenes, ejecutar el procesamiento y ver los resultados de la clasificación.
8.  **Puntos de Entrada:** El sistema puede ser ejecutado a través de `src/app_main.py` (menú principal) o `src/main.py` (flujo de trabajo principal).

Este flujo permite procesar imágenes, entrenar y utilizar modelos para la clasificación de caña de azúcar.

## Uso

Para ejecutar el sistema, puedes usar los siguientes comandos:

- **Menú principal (app_main.py):**
  ```bash
  python src/app_main.py
  ```
  Este script probablemente te presentará un menú con diferentes opciones para interactuar con el sistema.

- **Ejecución principal (main.py):**
  ```bash
  python src/main.py
  ```
  Este script podría ser el punto de entrada principal para una ejecución directa o un flujo de trabajo específico.


### Mantenimiento
Para limpiar archivos de caché Python:
```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
```
