import argparse
import os
import sys
import tkinter as tk

# Importaciones relativas al directorio actual
from model.training import train_model

# Importar la clase App directamente
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gui'))
from app import App

def setup_data_directories():
    """Crea las estructuras de directorios necesarias para el proyecto."""
    directories = [
        'data/model_training/train/ca',
        'data/model_training/train/no_ca',
        'data/model_training/val/ca',
        'data/model_training/val/no_ca',
        'data/raw',
        'data/processed',
        'checkpoints',
        'src/model/checkpoints'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def main():
    """Función principal que gestiona el entrenamiento del modelo y las interfaces gráficas."""
    parser = argparse.ArgumentParser(description='Sistema de Clasificación de Caña de Azúcar')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--clasificar', action='store_true', help='Iniciar la interfaz de clasificación')
    parser.add_argument('--bordes', action='store_true', help='Iniciar el detector de bordes')
    parser.add_argument('--medir', action='store_true', help='Iniciar la interfaz de medición de caña')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas para entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch para entrenamiento')
    
    args = parser.parse_args()
    
    # Asegurar que existan los directorios necesarios
    setup_data_directories()
    
    if args.train:
        print('Iniciando entrenamiento del modelo...')
        train_model(
            train_dir='data/model_training/train',
            val_dir='data/model_training/val',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        print('Entrenamiento completado.')
    
    if args.clasificar:
        print('Iniciando interfaz de clasificación...')
        # Crear y ejecutar la interfaz de clasificación directamente
        root = tk.Tk()
        app = App(root)
        root.mainloop()
        print('Clasificación completada.')
    elif args.bordes:
        print('Iniciando detector de bordes...')
        # Aquí debería ir la lógica para el detector de bordes o dejarlo como placeholder
        # app = DetectorBordesGUI()
        # app.iniciar()
        print('Funcionalidad de detector de bordes no implementada en este contexto.')
    elif args.medir:
        print('Iniciando interfaz de medición de caña...')
        # Importar la clase MedicionesGUI aquí para evitar problemas de importación circular
        from gui.mediciones_gui import MedicionesGUI
        # Crear la ventana principal de Tkinter
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal
        # Iniciar la interfaz de medición
        app = MedicionesGUI()
        print('Medición de caña completada.')
    elif not any([args.train, args.clasificar, args.bordes, args.medir]):
        # Si no se especifica ninguna opción, mostrar la interfaz de clasificación por defecto
        print('Iniciando interfaz de clasificación (por defecto)...')
        root = tk.Tk()
        app = App(root)
        root.mainloop()
        print('Clasificación completada.')

if __name__ == '__main__':
    main()