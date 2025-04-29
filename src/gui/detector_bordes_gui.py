import tkinter as tk
from tkinter import ttk, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor

class DetectorBordesGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖼️ Detector de Bordes")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#0d1117")
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None
        self.imagen_procesada = None
        self.modo_grises = tk.BooleanVar(value=False)
        self.modelo_cargado = False
        self.establecer_estilos()
        self.crear_interfaz()
        self.root.mainloop()

    def establecer_estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        base = "#0d1117"
        accent = "#2dffb3"
        text = "#e6e6ef"
        secondary_text = "#a1a1b5"
        line = "#30363d"
        estilo.configure("Switch.TCheckbutton", background=base, font=("Segoe UI", 11), foreground=accent)
        estilo.configure("TButton", font=("Segoe UI", 12, "bold"), padding=12, foreground=base, background=accent, borderwidth=0, relief="flat")
        estilo.map("TButton", background=[("active", accent), ("pressed", line)], foreground=[("active", base), ("pressed", accent)], relief=[("pressed", "groove")])
        estilo.configure("TLabel", background=base, font=("Segoe UI", 10), foreground=text)
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 18, "bold"), background=base, foreground=accent)
        estilo.configure("Secondary.TButton", font=("Segoe UI", 11), padding=8, foreground=accent, background=line)
        estilo.map("Secondary.TButton", background=[("active", accent)], relief=[("pressed", "sunken")])

    def crear_interfaz(self):
        # Barra superior personalizada con botones de ventana
        barra_superior = tk.Frame(self.root, bg="#0d1117", height=56)
        barra_superior.pack(side="top", fill="x")
        barra_superior.pack_propagate(False)
        # Botones de ventana
        btn_atras = tk.Label(barra_superior, text="\U0001F519", bg="#0d1117", fg="#2dffb3", font=("Segoe UI", 18, "bold"), cursor="hand2", width=4)
        btn_min = tk.Label(barra_superior, text="\u2796", bg="#0d1117", fg="#2dffb3", font=("Segoe UI", 18, "bold"), cursor="hand2", width=4)
        btn_close = tk.Label(barra_superior, text="\u274C", bg="#0d1117", fg="#2dffb3", font=("Segoe UI", 18, "bold"), cursor="hand2", width=4)
        btn_atras.pack(side="left", padx=(16,0), pady=8)
        btn_min.pack(side="right", padx=(0,0), pady=8)
        btn_close.pack(side="right", padx=(0,16), pady=8)
        # Animaciones hover
        for btn in [btn_atras, btn_min, btn_close]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#2dffb3", fg="#0d1117"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#0d1117", fg="#2dffb3"))
            btn.bind("<ButtonPress-1>", lambda e, b=btn: b.config(bg="#a1a1b5", fg="#0d1117"))
            btn.bind("<ButtonRelease-1>", lambda e, b=btn: b.config(bg="#2dffb3", fg="#0d1117"))
        # Funcionalidad de los botones
        btn_atras.bind("<Button-1>", lambda e: self.regresar_menu())
        btn_min.bind("<Button-1>", lambda e: self.root.iconify())
        btn_close.bind("<Button-1>", lambda e: self.root.destroy())
        # Canvas y scrollbar
        canvas = tk.Canvas(self.root, bg="#0d1117", highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main_frame = ttk.Frame(scrollable_frame, padding=0, style="TFrame")
        main_frame.pack(expand=True, fill='both')
        # Centrado vertical y horizontal
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        content_inner = ttk.Frame(main_frame, padding=40, style="TFrame")
        content_inner.grid(row=0, column=0, sticky="nsew")
        content_inner.columnconfigure(0, weight=1)
        titulo = ttk.Label(content_inner, text="\U0001F50D Detector de Bordes", style="Titulo.TLabel", anchor="center", justify="center")
        titulo.grid(row=0, column=0, pady=(30, 36), sticky="ew")
        frame_carga = ttk.Frame(content_inner, style="TFrame")
        frame_carga.grid(row=1, column=0, pady=16, sticky="ew")
        frame_carga.columnconfigure((0,1), weight=1)
        btn_cargar = ttk.Button(frame_carga, text="\U0001F4C2 Cargar Imagen", command=self.cargar_imagen, style="TButton")
        btn_cargar.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        btn_cargar_carpeta = ttk.Button(frame_carga, text="\U0001F4C1 Cargar Carpeta", command=self.cargar_carpeta, style="TButton")
        btn_cargar_carpeta.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.lbl_ruta = ttk.Label(content_inner, text="No se ha seleccionado ninguna imagen", wraplength=700, justify="center", style="TLabel")
        self.lbl_ruta.grid(row=2, column=0, pady=12, sticky="ew")
        self.switch_grises = ttk.Checkbutton(content_inner, text="\U0001F3A8 Modo grises", variable=self.modo_grises, style="Switch.TCheckbutton")
        self.switch_grises.grid(row=3, column=0, pady=12, sticky="ew")
        self.btn_procesar = ttk.Button(content_inner, text="\u2699\ufe0f Procesar Imagen", command=self.procesar_imagen, state='disabled', style="TButton")
        self.btn_procesar.grid(row=4, column=0, pady=12, sticky="ew")
        self.btn_probar_modelo = ttk.Button(content_inner, text="\U0001F9EA Probar Modelo", command=self.probar_modelo, state='disabled', style="TButton")
        self.btn_probar_modelo.grid(row=5, column=0, pady=12, sticky="ew")
        self.lbl_estado = ttk.Label(content_inner, text="", wraplength=700, foreground="#2dffb3", justify="center", style="TLabel")
        self.lbl_estado.grid(row=6, column=0, pady=12, sticky="ew")
        self.btn_regresar = ttk.Button(content_inner, text="Regresar", command=self.regresar_menu, style="Secondary.TButton")
        self.btn_regresar.grid(row=7, column=0, pady=16, sticky="ew")
        # Crear el frame para la vista previa
        self.frame_vista_previa = ttk.Frame(content_inner, style="TFrame")
        self.frame_vista_previa.grid(row=8, column=0, pady=16, sticky="nsew")
        self.frame_vista_previa.grid_remove()  # Inicialmente oculto
        
        # Modificación: asegurarnos de usar grid consistentemente
        self.frame_imagenes = ttk.Frame(self.frame_vista_previa, style="TFrame")
        self.frame_imagenes.grid(row=0, column=0, sticky="nsew")
        
        self.frame_original = ttk.Frame(self.frame_imagenes, style="TFrame")
        self.frame_original.grid(row=0, column=0, padx=16, sticky="nsew")
        ttk.Label(self.frame_original, text="Original", style="Titulo.TLabel").grid(row=0, column=0, pady=(0,12))
        self.lbl_original = ttk.Label(self.frame_original, style="TLabel")
        self.lbl_original.grid(row=1, column=0, pady=12)
        
        self.frame_procesada = ttk.Frame(self.frame_imagenes, style="TFrame")
        self.frame_procesada.grid(row=0, column=1, padx=16, sticky="nsew")
        ttk.Label(self.frame_procesada, text="Procesada", style="Titulo.TLabel").grid(row=0, column=0, pady=(0,12))
        self.lbl_vista_previa = ttk.Label(self.frame_procesada, style="TLabel")
        self.lbl_vista_previa.grid(row=1, column=0, pady=12)
        
        self.frame_botones_confirmacion = ttk.Frame(self.frame_vista_previa, style="TFrame")
        self.frame_botones_confirmacion.grid(row=1, column=0, pady=12)
        
        self.btn_aceptar = ttk.Button(self.frame_botones_confirmacion, text="\u2705 Aceptar", command=self.aceptar_imagen, style="TButton")
        self.btn_aceptar.grid(row=0, column=0, padx=16)
        
        self.btn_rechazar = ttk.Button(self.frame_botones_confirmacion, text="\u274C Rechazar", command=self.rechazar_imagen, style="TButton")
        self.btn_rechazar.grid(row=0, column=1, padx=16)

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

    def cargar_carpeta(self):
        """
        Permite al usuario seleccionar una carpeta y procesar todas las imágenes en ella.
        """
        try:
            # Solicitar al usuario la carpeta
            ruta_carpeta = self.file_manager.seleccionar_carpeta()
            
            if not ruta_carpeta:
                return  # El usuario canceló la selección
            
            # Obtener lista de imágenes en la carpeta
            rutas_imagenes = self.file_manager.obtener_imagenes_de_carpeta(ruta_carpeta)
            
            if not rutas_imagenes:
                messagebox.showinfo("Información", "No se encontraron imágenes en la carpeta seleccionada.")
                return
            
            # Preguntar al usuario si desea proceder
            respuesta = messagebox.askyesno(
                "Procesar carpeta", 
                f"Se encontraron {len(rutas_imagenes)} imágenes en la carpeta.\n¿Desea procesar todas las imágenes?"
            )
            
            if not respuesta:
                return
            
            # Preguntar si desea convertir a escala de grises
            aplicar_grises = messagebox.askyesno(
                "Modo grises", 
                "¿Desea aplicar el modo escala de grises a todas las imágenes procesadas?"
            )
            
            # Definir función de procesamiento
            def procesar_funcion(ruta_imagen):
                try:
                    resultado = self.image_processor.procesar_imagen_completa(ruta_imagen)
                    imagen_procesada = resultado['imagen_A4'].copy()
                    
                    # Aplicar escala de grises si es necesario
                    if aplicar_grises:
                        from img_proc.esc_grises import convertir_a_grises
                        imagen_procesada = convertir_a_grises(imagen_procesada)
                    
                    return imagen_procesada
                except Exception as e:
                    print(f"Error al procesar {ruta_imagen}: {e}")
                    raise
            
            # Iniciar procesamiento
            self.lbl_estado.config(text=f"⏳ Procesando {len(rutas_imagenes)} imágenes...")
            self.file_manager.procesar_carpeta(ruta_carpeta, procesar_funcion)
            
            # Actualizar estado
            self.lbl_estado.config(text=f"✅ Procesamiento de carpeta completado.")
            
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la carpeta:\n{str(e)}")

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
            self.btn_probar_modelo.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la imagen:\n{str(e)}")

    def probar_modelo(self):
        try:
            if not self.modelo_cargado:
                from model.deep_sugarcane_model import cargar_modelo
                self.modelo = cargar_modelo()
                self.modelo_cargado = True
            
            predicciones = self.modelo.predecir(self.imagen_procesada)
            self.mostrar_resultados(predicciones)
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al probar el modelo:\n{str(e)}")

    def mostrar_resultados(self, predicciones):
        # Crear ventana para mostrar resultados
        ventana_resultados = tk.Toplevel(self.root)
        ventana_resultados.title("Resultados del Modelo")
        ventana_resultados.geometry("400x300")
        
        # Mostrar medidas
        frame_medidas = ttk.Frame(ventana_resultados, padding=20)
        frame_medidas.pack(expand=True, fill='both')
        
        for i, medida in enumerate(predicciones):
            ttk.Label(frame_medidas, text=f"Medida {i+1}: {medida:.2f} cm").pack(anchor='w')
        
        # Botón para cerrar
        ttk.Button(ventana_resultados, text="Cerrar", command=ventana_resultados.destroy).pack(pady=10)

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
        
        # Mostrar la imagen original
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
        
        # Mostrar la imagen procesada
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
        
        # Mostrar el marco de vista previa
        self.frame_vista_previa.grid()  # Usar grid en lugar de pack
        self.lbl_estado.config(text="✅ Imagen procesada. Confirme para guardar.")
        
        # ELIMINADO: este botón estaba usando pack en lugar de grid
        # btn_back = tk.Button(self.frame_vista_previa, text="Regresar", command=self.regresar_menu, font=("Segoe UI", 12, "bold"), bg="#64748B", fg="#FFF")
        # btn_back.pack(pady=10)  # esto causa el conflicto de gestores de geometría

    def limpiar_estado(self):
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None
        self.imagen_procesada = None
        self.modo_grises.set(False)
        self.btn_procesar.config(state='disabled')
        self.lbl_ruta.config(text="No se ha seleccionado ninguna imagen")
        self.switch_grises.config(state='enable')
        self.frame_vista_previa.grid_remove()  # Usar grid_remove en lugar de pack_forget

    def aceptar_imagen(self):
        self.confirmar_guardado(self.info_imagen_raw, self.info_resultado)
        self.limpiar_estado()

    def rechazar_imagen(self):
        self.cancelar_guardado()
        self.limpiar_estado()

    def regresar_menu(self):
        self.root.destroy()
        import sys
        import os
        import subprocess
        main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app_main.py'))
        subprocess.Popen([sys.executable, main_path])