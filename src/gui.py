import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor

class DetectorBordesGUI:
    """Clase que maneja la interfaz gráfica del detector de bordes (moderna)."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖼️ Detector de Bordes")
        self.root.geometry("640x480")
        self.root.configure(bg="#F1F6F9")

        # Inicializar manejadores
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()

        # Variables
        self.imagen_path = None

        # Estilo moderno
        self.establecer_estilos()

        # Crear interfaz
        self.crear_interfaz()

    def establecer_estilos(self):
        """Configura estilos modernos para ttk."""
        estilo = ttk.Style()
        estilo.theme_use("clam")

        color_primario = "#112D4E"  # Azul oscuro
        color_secundario = "#1E40AF"  # Azul vibrante
        fondo = "#F1F6F9"
        blanco = "#FFFFFF"

        estilo.configure("TButton",
                         font=("Segoe UI", 11, "bold"),
                         padding=10,
                         foreground=blanco,
                         background=color_secundario,
                         borderwidth=0)
        estilo.map("TButton",
                   background=[("active", color_primario)],
                   relief=[("pressed", "sunken")])

        estilo.configure("TLabel", background=fondo, font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 18, "bold"), background=fondo, foreground=color_primario)

    def crear_interfaz(self):
        """Crea la interfaz gráfica de usuario."""
        main_frame = ttk.Frame(self.root, padding=30)
        main_frame.pack(expand=True, fill='both')

        # Título
        titulo = ttk.Label(main_frame, text="🔍 Detector de Bordes", style="Titulo.TLabel")
        titulo.pack(pady=(10, 25))

        # Botón para cargar imagen
        btn_cargar = ttk.Button(main_frame, text="📂 Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=10, fill='x')

        # Etiqueta para mostrar ruta
        self.lbl_ruta = ttk.Label(main_frame, text="No se ha seleccionado ninguna imagen",
                                  wraplength=560, justify="center")
        self.lbl_ruta.pack(pady=10)

        # Botón para procesar
        self.btn_procesar = ttk.Button(main_frame, text="⚙️ Procesar Imagen", command=self.procesar_imagen, state='disabled')
        self.btn_procesar.pack(pady=10, fill='x')

        # Etiqueta para mostrar estado
        self.lbl_estado = ttk.Label(main_frame, text="", wraplength=560,
                                    foreground="#1E40AF", justify="center")
        self.lbl_estado.pack(pady=10)

    def cargar_imagen(self):
        """Maneja la carga de imágenes desde el sistema de archivos."""
        try:
            archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
            if archivo:
                self.imagen_path = archivo
                nombre_archivo = self.file_manager.guardar_imagen_raw(archivo)

                self.lbl_ruta.config(text=f"🖼️ Imagen cargada: {nombre_archivo}")
                self.btn_procesar.config(state='normal')
                self.lbl_estado.config(text="✅ Imagen guardada en `data/raw`")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al cargar la imagen:\n{str(e)}")

    def procesar_imagen(self):
        """Procesa la imagen seleccionada y muestra resultados."""
        try:
            if not self.imagen_path:
                raise ValueError("No se ha seleccionado ninguna imagen")

            resultado = self.image_processor.procesar_imagen_completa(self.imagen_path)

            nombre_archivo = self.file_manager.guardar_resultados(
                resultado['imagen_A4'],
                self.imagen_path
            )

            self.lbl_estado.config(text=f"📁 Imagen guardada como:\n{nombre_archivo}")
            messagebox.showinfo("Éxito", "🎉 Imagen procesada y guardada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la imagen:\n{str(e)}")

    def iniciar(self):
        """Inicia el loop principal de la interfaz gráfica."""
        self.root.mainloop()