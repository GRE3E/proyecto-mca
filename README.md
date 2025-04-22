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

## Requisitos adicionales

Este proyecto utiliza [Git Large File Storage (LFS)](https://git-lfs.github.com) para manejar archivos grandes (como modelos `.pth`).

## Instalación

0. Instala Git LFS (solo una vez por máquina):
   ```bash
   git lfs install

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
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/          # Modelos entrenados
├── data/
│   ├── model_training/   # Datos para entrenamiento
│   │   ├── train/       # Conjunto de entrenamiento
│   │   │   ├── ca/      # Imágenes de caña
│   │   │   └── no_ca/   # Imágenes que no son caña
│   │   └── val/         # Conjunto de validación
│   │       ├── ca/
│   │       └── no_ca/
│   ├── processed/       # Imágenes procesadas
│   └── raw/            # Imágenes sin procesar
└── src/
    ├── app_main.py      # Menú principal (punto de entrada)
    ├── img_proc/        # Procesamiento de imágenes
    │   ├── edge_detection.py
    │   ├── esc_grises.py
    │   ├── main_processor.py
    │   └── roi_extraction.py
    ├── model/           # Modelos de ML
    │   ├── inference.py
    │   ├── training.py
    │   └── yolo_model.py
    ├── gui/             # Interfaces gráficas
    │   └── app.py
    ├── file_manager.py
    ├── main.py
    └── utils.py
```

## Uso

### Menú Principal
Para iniciar el menú principal:
```bash
python src/app_main.py
```
Desde aquí puedes acceder a:
- Reducción de bordes
- Entrenamiento del modelo
- Prueba del modelo

**Nota:** El modelo entrenado `best_sugarcane_model.pth` se encuentra incluido en el repositorio en la carpeta `src/model/`. Puedes usarlo directamente para probar la funcionalidad desde el menú principal.
### Entrenamiento del Modelo
Desde el menú, selecciona "Entrenamiento Modelo". Se mostrará información sobre las rutas de los datos, cantidad de archivos y validación. El proceso maneja excepciones y muestra mensajes claros en caso de error.

### Clasificación de Imágenes
Selecciona "Probar modelo" en el menú. Podrás subir una imagen nueva y el sistema indicará si es o no caña de azúcar.

### Detector de Bordes
Selecciona "Bordes" en el menú para acceder a la reducción de bordes usando los módulos de procesamiento de imágenes.

### Mantenimiento
Para limpiar archivos de caché Python:
```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
```