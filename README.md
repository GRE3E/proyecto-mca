
# Proyecto de Detección de Cañas de Azúcar

## Configuración del Proyecto

### Prerrequisitos
- Python 3.10
- Git instalado en tu sistema
- Pip (gestor de paquetes de Python)

### Clonar el Repositorio

1. Abre una terminal o línea de comandos
2. Clona el repositorio usando Git:
   ```bash
   git clone https://github.com/GR3E3/deteccion_cana_azucar.git
   cd deteccion_cana_azucar
   ```
3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Ejecuta el proyecto principal:
   ```
   python src\Detencion.py
   ```

### Instalación de Dependencias

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Estructura del Proyecto

- `/src`                   - Código fuente principal
  - `Detencion.py`         - Módulo de detección
  - `interfaz.py`          - Interfaz de usuario
  - `medidas.py`           - Módulo de mediciones
- `/output`                - Resultados y archivos generados
  - `/models`              - Modelos entrenados
  - `/predictions`         - Predicciones realizadas
  - `/logs`                - Archivos de registro
- `requirements.txt`       - Lista de dependencias
- `README.md`             - Este archivo
- `.gitignore`            - Configuración de Git

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
