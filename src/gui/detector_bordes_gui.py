import tkinter as tk
from tkinter import ttk, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor

class DetectorBordesGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖼️ Detector de Bordes")
        ancho_pantalla = self.root.winfo_screenwidth()
        alto_pantalla = self.root.winfo_screenheight()
        self.root.geometry(f"{ancho_pantalla}x{alto_pantalla}+0+0")
        self.root.state('zoomed')
        self.root.configure(bg="#F1F6F9")
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None
        self.imagen_procesada = None
        self.modo_grises = tk.BooleanVar(value=False)
        self.establecer_estilos()
        self.crear_interfaz()
        self.root.mainloop()

    def establecer_estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        color_primario = "#112D4E"
        color_secundario = "#1E40AF"
        fondo = "#F1F6F9"
        blanco = "#FFFFFF"
        estilo.configure("Switch.TCheckbutton", background=fondo, font=("Segoe UI", 11), foreground=color_primario)
        estilo.configure("TButton", font=("Segoe UI", 11, "bold"), padding=10, foreground=blanco, background=color_secundario, borderwidth=0)
        estilo.map("TButton", background=[("active", color_primario)], relief=[("pressed", "sunken")])
        estilo.configure("TLabel", background=fondo, font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 18, "bold"), background=fondo, foreground=color_primario)

    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding=30)
        main_frame.pack(expand=True, fill='both')
        titulo = ttk.Label(main_frame, text="🔍 Detector de Bordes", style="Titulo.TLabel")
        titulo.pack(pady=(10, 25))
        btn_cargar = ttk.Button(main_frame, text="📂 Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=10, fill='x')
        self.lbl_ruta = ttk.Label(main_frame, text="No se ha seleccionado ninguna imagen", wraplength=560, justify="center")
        self.lbl_ruta.pack(pady=10)
        self.switch_grises = ttk.Checkbutton(main_frame, text="🎨 Modo grises", variable=self.modo_grises, style="Switch.TCheckbutton")
        self.switch_grises.pack(pady=10)
        self.btn_procesar = ttk.Button(main_frame, text="⚙️ Procesar Imagen", command=self.procesar_imagen, state='disabled')
        self.btn_procesar.pack(pady=10, fill='x')
        self.lbl_estado = ttk.Label(main_frame, text="", wraplength=560, foreground="#1E40AF", justify="center")
        self.lbl_estado.pack(pady=10)
        self.frame_vista_previa = ttk.Frame(main_frame, padding=10)
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        self.frame_vista_previa.pack_forget()
        self.frame_imagenes = ttk.Frame(self.frame_vista_previa)
        self.frame_imagenes.pack(fill='both', expand=True)
        self.frame_original = ttk.Frame(self.frame_imagenes)
        self.frame_original.pack(side=tk.LEFT, padx=10, fill='both', expand=True)
        ttk.Label(self.frame_original, text="Original", style="Titulo.TLabel").pack(pady=(0,10))
        self.lbl_original = ttk.Label(self.frame_original)
        self.lbl_original.pack(pady=10)
        self.frame_procesada = ttk.Frame(self.frame_imagenes)
        self.frame_procesada.pack(side=tk.LEFT, padx=10, fill='both', expand=True)
        ttk.Label(self.frame_procesada, text="Procesada", style="Titulo.TLabel").pack(pady=(0,10))
        self.lbl_vista_previa = ttk.Label(self.frame_procesada)
        self.lbl_vista_previa.pack(pady=10)
        self.frame_botones_confirmacion = ttk.Frame(self.frame_vista_previa, padding=10)
        self.frame_botones_confirmacion.pack(pady=10)
        self.btn_aceptar = ttk.Button(self.frame_botones_confirmacion, text="✅ Aceptar", command=self.aceptar_imagen)
        self.btn_aceptar.pack(side=tk.LEFT, padx=10)
        self.btn_rechazar = ttk.Button(self.frame_botones_confirmacion, text="❌ Rechazar", command=self.rechazar_imagen)
        self.btn_rechazar.pack(side=tk.LEFT, padx=10)

    def cargar_imagen(self):
        from tkinter import filedialog
        try:
            archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
            if archivo:
                self.imagen_path = archivo
                self.info_imagen_raw = self.file_manager.preparar_imagen_raw(archivo)
                self.lbl_ruta.config(text=f"🖼️ Imagen seleccionada: {self.info_imagen_raw['nuevo_nombre']}")
                self.btn_procesar.config(state='normal')
                self.lbl_estado.config(text="✅ Imagen lista para procesar")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al cargar la imagen:\n{str(e)}")

    def procesar_imagen(self):
        try:
            if not self.imagen_path or not hasattr(self, 'info_imagen_raw'):
                return
            resultado = self.image_processor.procesar_imagen_completa(self.imagen_path)
            self.imagen_procesada = resultado['imagen_A4'].copy()
            self.info_resultado = self.file_manager.preparar_resultados(resultado['imagen_A4'], self.imagen_path)
            self.switch_grises.config(state='active')
            self.modo_grises.trace_add('write', lambda *args: self.actualizar_vista_previa())
            self.mostrar_vista_previa_integrada(self.imagen_procesada)
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la imagen:\n{str(e)}")

    def confirmar_guardado(self, info_imagen_raw, info_resultado):
        try:
            nombre_raw = self.file_manager.guardar_imagen_raw(info_imagen_raw)
            if self.modo_grises.get():
                from img_proc.esc_grises import convertir_a_grises
                imagen_a_guardar = convertir_a_grises(info_resultado['imagen'])
                info_resultado['imagen'] = imagen_a_guardar
            nombre_procesado = self.file_manager.guardar_resultados(info_resultado)
            self.lbl_estado.config(text=f"📁 Imágenes guardadas correctamente:\n{nombre_procesado}")
            messagebox.showinfo("Éxito", "🎉 Imagen procesada y guardada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al guardar las imágenes:\n{str(e)}")

    def cancelar_guardado(self):
        self.lbl_estado.config(text="❌ Proceso cancelado por el usuario")
        messagebox.showinfo("Cancelado", "El proceso ha sido cancelado. No se han guardado imágenes.")
        self.imagen_procesada = None
        self.modo_grises.set(False)
        self.switch_grises.config(state='enable')

    def actualizar_vista_previa(self):
        if self.imagen_procesada is not None:
            self.mostrar_vista_previa_integrada(self.imagen_procesada)

    def mostrar_vista_previa_integrada(self, imagen):
        import cv2
        import numpy as np
        from PIL import Image, ImageTk
        imagen_original = cv2.imread(self.imagen_path)
        imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
        alto_orig, ancho_orig = imagen_original_rgb.shape[:2]
        max_size = 400
        if alto_orig > max_size or ancho_orig > max_size:
            if alto_orig > ancho_orig:
                nuevo_alto = max_size
                nuevo_ancho = int(ancho_orig * (max_size / alto_orig))
            else:
                nuevo_ancho = max_size
                nuevo_alto = int(alto_orig * (max_size / ancho_orig))
            imagen_original_rgb = cv2.resize(imagen_original_rgb, (nuevo_ancho, nuevo_alto))
        imagen_original_pil = Image.fromarray(imagen_original_rgb)
        imagen_original_tk = ImageTk.PhotoImage(imagen_original_pil)
        self.lbl_original.config(image=imagen_original_tk)
        self.lbl_original.image = imagen_original_tk
        if isinstance(imagen, np.ndarray):
            imagen_mostrar = imagen.copy()
            if self.modo_grises.get():
                from img_proc.esc_grises import convertir_a_grises
                imagen_mostrar = convertir_a_grises(imagen_mostrar)
                imagen_rgb = cv2.cvtColor(imagen_mostrar, cv2.COLOR_GRAY2RGB)
            else:
                if len(imagen_mostrar.shape) == 3 and imagen_mostrar.shape[2] == 3:
                    imagen_rgb = cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB)
                else:
                    imagen_rgb = imagen_mostrar
            alto, ancho = imagen_rgb.shape[:2]
            if alto > max_size or ancho > max_size:
                if alto > ancho:
                    nuevo_alto = max_size
                    nuevo_ancho = int(ancho * (max_size / alto))
                else:
                    nuevo_ancho = max_size
                    nuevo_alto = int(alto * (max_size / ancho))
                imagen_rgb = cv2.resize(imagen_rgb, (nuevo_ancho, nuevo_alto))
            imagen_pil = Image.fromarray(imagen_rgb)
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
        else:
            imagen_tk = ImageTk.PhotoImage(imagen)
        self.lbl_vista_previa.config(image=imagen_tk)
        self.lbl_vista_previa.image = imagen_tk
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        self.lbl_estado.config(text="✅ Imagen procesada. Confirme para guardar.")

    def limpiar_estado(self):
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None
        self.imagen_procesada = None
        self.modo_grises.set(False)
        self.btn_procesar.config(state='disabled')
        self.lbl_ruta.config(text="No se ha seleccionado ninguna imagen")
        self.switch_grises.config(state='enable')
        self.frame_vista_previa.pack_forget()

    def aceptar_imagen(self):
        self.confirmar_guardado(self.info_imagen_raw, self.info_resultado)
        self.limpiar_estado()

    def rechazar_imagen(self):
        self.cancelar_guardado()
        self.limpiar_estado()
        self.frame_vista_previa.pack_forget()