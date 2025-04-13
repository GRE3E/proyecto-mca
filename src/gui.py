import tkinter as tk
from tkinter import filedialog, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor

class DetectorBordesGUI:
    """Clase que maneja la interfaz gráfica del detector de bordes."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detector de Bordes")
        self.root.geometry("600x400")
        
        # Inicializar manejadores
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()
        
        # Variables
        self.imagen_path = None
        
        # Crear interfaz
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica de usuario."""
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Título
        titulo = tk.Label(main_frame, text="Detector de Bordes", font=("Arial", 16, "bold"))
        titulo.pack(pady=10)
        
        # Botón para cargar imagen
        btn_cargar = tk.Button(main_frame, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=10)
        
        # Etiqueta para mostrar ruta de la imagen
        self.lbl_ruta = tk.Label(main_frame, text="No se ha seleccionado ninguna imagen", wraplength=500)
        self.lbl_ruta.pack(pady=10)
        
        # Botón para procesar imagen
        self.btn_procesar = tk.Button(main_frame, text="Procesar Imagen", command=self.procesar_imagen, state='disabled')
        self.btn_procesar.pack(pady=10)
        
        # Etiqueta para mostrar estado
        self.lbl_estado = tk.Label(main_frame, text="")
        self.lbl_estado.pack(pady=10)
    
    def cargar_imagen(self):
        """Maneja la carga de imágenes desde el sistema de archivos."""
        try:
            archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
            if archivo:
                self.imagen_path = archivo
                nombre_archivo = self.file_manager.guardar_imagen_raw(archivo)
                
                self.lbl_ruta.config(text=f"Imagen cargada: {nombre_archivo}")
                self.btn_procesar.config(state='normal')
                self.lbl_estado.config(text="Imagen guardada en data/raw")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")
    
    def procesar_imagen(self):
        """Procesa la imagen seleccionada y muestra resultados."""
        try:
            if not self.imagen_path:
                raise ValueError("No se ha seleccionado ninguna imagen")
            
            # Procesar imagen y obtener resultados
            resultado = self.image_processor.procesar_imagen_completa(self.imagen_path)
            
            # Guardar resultados
            nombres_archivos = self.file_manager.guardar_resultados(
                resultado['imagen_A4'], 
                resultado['imagen_bordes'], 
                resultado['imagen_mediciones'],
                self.imagen_path
            )
            
            # Actualizar interfaz
            self.lbl_estado.config(text=f"Imágenes guardadas como:\n{nombres_archivos[0]}\n{nombres_archivos[1]}\n{nombres_archivos[2]}")
            messagebox.showinfo("Éxito", "Imágenes procesadas y guardadas correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
    
    def iniciar(self):
        """Inicia el loop principal de la interfaz gráfica."""
        self.root.mainloop()