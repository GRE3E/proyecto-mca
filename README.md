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
- RE Entrenamiento

**Nota:** El modelo entrenado `best_sugarcane_model.pth` se encuentra incluido en el repositorio en la carpeta `src/model/`. Puedes usarlo directamente para probar la funcionalidad desde el menú principal.

### Flujo de Procesamiento de Imágenes
1. **Reducción de bordes:** Antes de la predicción, las imágenes pasan por un proceso de reducción de bordes para mejorar la detección y clasificación.
2. **Clasificación:** El modelo clasifica la imagen procesada como caña o no caña.
3. **Visualización:** Puedes ver la imagen original y la procesada, y decidir si guardar el resultado.

### Entrenamiento del Modelo
Desde el menú, selecciona "Entrenamiento Modelo". Se abrirá una ventana emergente donde se muestran los logs del proceso en tiempo real. El proceso maneja excepciones y muestra mensajes claros en caso de error.

### RE Entrenamiento (Nuevo)
Permite seleccionar un modelo existente mediante el gestor de archivos y reentrenarlo con nuevos datos. El progreso y los logs se muestran en una ventana emergente. Incluye botón de regreso para volver al menú principal en cualquier momento.

### Clasificación de Imágenes
Selecciona "Probar modelo" en el menú. Podrás subir una imagen nueva y el sistema indicará si es o no caña de azúcar. El flujo incluye la reducción de bordes antes de la predicción.

### Detector de Bordes
Selecciona "Bordes" en el menú para acceder a la reducción de bordes usando los módulos de procesamiento de imágenes. Puedes visualizar la imagen original y la procesada, y guardar el resultado si lo deseas. Incluye botón de regreso siempre visible.

### Visualización de Logs
Tanto el entrenamiento como el reentrenamiento muestran los logs en ventanas emergentes, permitiendo un seguimiento detallado del proceso.

### Mantenimiento
Para limpiar archivos de caché Python:
```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
```