import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor
import cv2
import numpy as np
from PIL import Image, ImageTk

class DetectorBordesGUI:
    """Clase que maneja la interfaz gráfica del detector de bordes (moderna)."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖼️ Detector de Bordes")
        self.root.geometry("1080x920")
        self.root.configure(bg="#F1F6F9")

        # Inicializar manejadores
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()

        # Variables
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None

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
        
        # Área para vista previa de imagen
        self.frame_vista_previa = ttk.Frame(main_frame, padding=10)
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        self.frame_vista_previa.pack_forget()  # Inicialmente oculto
        
        # Etiqueta para la imagen de vista previa
        self.lbl_vista_previa = ttk.Label(self.frame_vista_previa)
        self.lbl_vista_previa.pack(pady=10)
        
        # Frame para botones de confirmación
        self.frame_botones_confirmacion = ttk.Frame(self.frame_vista_previa, padding=10)
        self.frame_botones_confirmacion.pack(pady=10)
        
        # Botones de aceptar y rechazar
        self.btn_aceptar = ttk.Button(self.frame_botones_confirmacion, text="✅ Aceptar", command=self.aceptar_imagen)
        self.btn_aceptar.pack(side=tk.LEFT, padx=10)
        
        self.btn_rechazar = ttk.Button(self.frame_botones_confirmacion, text="❌ Rechazar", command=self.rechazar_imagen)
        self.btn_rechazar.pack(side=tk.LEFT, padx=10)

    def cargar_imagen(self):
        """Maneja la carga de imágenes desde el sistema de archivos."""
        try:
            archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
            if archivo:
                self.imagen_path = archivo
                # Solo preparamos la información, no guardamos aún
                self.info_imagen_raw = self.file_manager.preparar_imagen_raw(archivo)

                self.lbl_ruta.config(text=f"🖼️ Imagen seleccionada: {self.info_imagen_raw['nuevo_nombre']}")
                self.btn_procesar.config(state='normal')
                self.lbl_estado.config(text="✅ Imagen lista para procesar")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al cargar la imagen:\n{str(e)}")

    def procesar_imagen(self):
        """Procesa la imagen seleccionada y muestra vista previa para confirmar guardado."""
        try:
            if not self.imagen_path:
                raise ValueError("No se ha seleccionado ninguna imagen")

            resultado = self.image_processor.procesar_imagen_completa(self.imagen_path)
            
            # Preparar información para guardar resultados, pero no guardar aún
            self.info_resultado = self.file_manager.preparar_resultados(
                resultado['imagen_A4'],
                self.imagen_path
            )
            
            # Mostrar vista previa en la interfaz principal
            self.mostrar_vista_previa_integrada(resultado['imagen_A4'])
            
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la imagen:\n{str(e)}")

    def confirmar_guardado(self, info_imagen_raw, info_resultado):
        """Guarda tanto la imagen original como la procesada."""
        try:
            # Guardar la imagen original en raw
            nombre_raw = self.file_manager.guardar_imagen_raw(info_imagen_raw)
            
            # Guardar la imagen procesada
            nombre_procesado = self.file_manager.guardar_resultados(info_resultado)
            
            self.lbl_estado.config(text=f"📁 Imágenes guardadas correctamente:\n{nombre_procesado}")
            messagebox.showinfo("Éxito", "🎉 Imagen procesada y guardada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al guardar las imágenes:\n{str(e)}")
    
    def cancelar_guardado(self):
        """Cancela el proceso de guardado."""
        self.lbl_estado.config(text="❌ Proceso cancelado por el usuario")
        messagebox.showinfo("Cancelado", "El proceso ha sido cancelado. No se han guardado imágenes.")
        
    def mostrar_vista_previa_integrada(self, imagen):
        """Muestra la vista previa de la imagen procesada en la interfaz principal."""
        # Convertir imagen de OpenCV a formato para tkinter
        if isinstance(imagen, np.ndarray):
            # Convertir de BGR a RGB
            if len(imagen.shape) == 3 and imagen.shape[2] == 3:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen_rgb = imagen
                
            # Redimensionar si es muy grande
            alto, ancho = imagen_rgb.shape[:2]
            max_size = 400  # Tamaño máximo para la vista previa integrada
            if alto > max_size or ancho > max_size:
                if alto > ancho:
                    nuevo_alto = max_size
                    nuevo_ancho = int(ancho * (max_size / alto))
                else:
                    nuevo_ancho = max_size
                    nuevo_alto = int(alto * (max_size / ancho))
                imagen_rgb = cv2.resize(imagen_rgb, (nuevo_ancho, nuevo_alto))
            
            # Convertir a formato PIL y luego a PhotoImage
            imagen_pil = Image.fromarray(imagen_rgb)
            from PIL import ImageTk
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
        else:
            # Si ya es un objeto PIL Image
            imagen_tk = ImageTk.PhotoImage(imagen)
        
        # Mostrar imagen en la etiqueta de vista previa
        self.lbl_vista_previa.config(image=imagen_tk)
        self.lbl_vista_previa.image = imagen_tk  # Mantener referencia
        
        # Mostrar el frame de vista previa
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        
        # Actualizar estado
        self.lbl_estado.config(text="✅ Imagen procesada. Confirme para guardar.")
    
    def aceptar_imagen(self):
        """Acepta la imagen procesada y la guarda."""
        self.confirmar_guardado(self.info_imagen_raw, self.info_resultado)
        self.frame_vista_previa.pack_forget()  # Ocultar vista previa
    
    def rechazar_imagen(self):
        """Rechaza la imagen procesada."""
        self.cancelar_guardado()
        self.frame_vista_previa.pack_forget()  # Ocultar vista previa
    
    def iniciar(self):
        """Inicia el loop principal de la interfaz gráfica."""
        self.root.mainloop()