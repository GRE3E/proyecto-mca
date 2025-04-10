# Cana Azúcar Detector

Proyecto para detectar cañas de azúcar en imágenes y extraer características morfológicas.


PYTHON Versión 10.0

/deteccion_cana_azucar
├── /data                  # Carpeta para los datos (imágenes)
│   ├── /raw               # Imágenes originales sin procesar
│   └── /processed         # Imágenes preprocesadas
│
├── /notebooks             # Notebooks de Jupyter para exploración, análisis y pruebas
│
├── /src                   # Código fuente del proyecto
│   ├── /preprocessing     # Scripts para preprocesamiento de imágenes
│   ├── /feature_extraction # Scripts para la extracción de características
│   ├── /model             # Scripts para la definición, entrenamiento y evaluación del modelo
│   └── /utils             # Funciones auxiliares y utilitarias (por ejemplo, para visualización)
│
├── /tests                 # Pruebas automatizadas para las funciones y modelos
│
├── /output                # Resultados, modelos entrenados y archivos generados
│   ├── /models            # Modelos entrenados
│   ├── /predictions       # Predicciones realizadas con el modelo
│   └── /logs              # Archivos de registro (si es necesario)
│
├── /web                   # Código para la interfaz web
│   ├── /static            # Archivos estáticos como imágenes, CSS, JS
│   └── /templates         # Plantillas HTML
│
├── requirements.txt       # Lista de dependencias de Python
├── README.md              # Descripción del proyecto, instrucciones, etc.
├── .gitignore             # Archivos y carpetas que Git debe ignorar
└── config.py              # Archivo de configuración (si es necesario)
