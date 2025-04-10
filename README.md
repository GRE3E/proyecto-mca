# Cana Azúcar Detector

Proyecto para detectar cañas de azúcar en imágenes y extraer características morfológicas.


PYTHON Versión 10.0

# Proyecto de Cañas de Azúcar

Este proyecto tiene como objetivo desarrollar un programa que detecte cañas de azúcar en imágenes y extraiga sus características. El proyecto utiliza Python, OpenCV y otras librerías necesarias para preprocesar imágenes, entrenar un modelo y hacer predicciones.

## Estructura del Proyecto

La estructura del proyecto es la siguiente:

- `/deteccion_cana_azucar`
  - `/data`                  - Carpeta para los datos (imágenes)
    - `/raw`                 - Imágenes originales sin procesar
    - `/processed`           - Imágenes preprocesadas
  - `/notebooks`             - Notebooks de Jupyter para exploración, análisis y pruebas
  - `/src`                   - Código fuente del proyecto
    - `/preprocessing`       - Scripts para preprocesamiento de imágenes
    - `/feature_extraction` - Scripts para la extracción de características
    - `/model`               - Scripts para la definición, entrenamiento y evaluación del modelo
    - `/utils`               - Funciones auxiliares y utilitarias (por ejemplo, para visualización)
  - `/tests`                 - Pruebas automatizadas para las funciones y modelos
  - `/output`                - Resultados, modelos entrenados y archivos generados
    - `/models`              - Modelos entrenados
    - `/predictions`         - Predicciones realizadas con el modelo
    - `/logs`                - Archivos de registro (si es necesario)
  - `/web`                   - Código para la interfaz web
    - `/static`              - Archivos estáticos como imágenes, CSS, JS
    - `/templates`           - Plantillas HTML
  - `requirements.txt`       - Lista de dependencias de Python
  - `README.md`              - Descripción del proyecto, instrucciones, etc.
  - `.gitignore`             - Archivos y carpetas que Git debe ignorar
  - `config.py`              - Archivo de configuración (si es necesario)

### Descripción de las carpetas:

- **/data**: Aquí se almacenan todos los datos, tanto los originales como los preprocesados.
- **/notebooks**: Carpeta para almacenar notebooks de Jupyter, donde podrás realizar experimentos y análisis exploratorios.
- **/src**: El código fuente principal del proyecto. Aquí divides tus scripts en diferentes subcarpetas para tareas específicas como preprocesamiento, extracción de características y el modelo.
- **/tests**: Aquí van las pruebas para asegurarte de que cada parte del proyecto funciona correctamente (pueden ser pruebas unitarias o de integración).
- **/output**: Carpeta para almacenar los resultados generados, como los modelos entrenados, las predicciones y los logs.
- **/web**: Si estás creando una interfaz web para cargar imágenes, aquí guardas todo el código relacionado (por ejemplo, en Flask o Django).
- **requirements.txt**: Un archivo donde listarás las librerías y dependencias de Python para que cualquier persona pueda instalar lo que se necesita para ejecutar el proyecto.
- **README.md**: Un archivo de texto donde describes el proyecto, cómo configurarlo, cómo ejecutarlo y cualquier otra información útil.
- **.gitignore**: Archivo que le dice a Git qué archivos o carpetas no debe rastrear (como archivos temporales o datos grandes que no deben subirse al repositorio).
