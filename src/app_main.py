import tkinter as tk
from tkinter import ttk, messagebox
from file_manager import FileManager
from img_proc.main_processor import ImageProcessor
import os
import subprocess

class MainInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Menú Principal - Proyecto MCA")
        self.root.geometry("1080x720")
        self.root.configure(bg="#F1F6F9")
        self.establecer_estilos()
        self.crear_interfaz()
        self.root.mainloop()

    def establecer_estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#FFFFFF", background="#1E40AF", borderwidth=0)
        estilo.map("TButton", background=[("active", "#3B82F6")])

    def crear_interfaz(self):
        frame = ttk.Frame(self.root, padding=40)
        frame.pack(expand=True)
        label = ttk.Label(frame, text="Seleccione una opción:", font=("Segoe UI", 16, "bold"))
        label.pack(pady=20)
        btn_bordes = ttk.Button(frame, text="Bordes", command=self.abrir_bordes)
        btn_bordes.pack(fill="x", pady=10)
        btn_entrenar = ttk.Button(frame, text="Entrenamiento Modelo", command=self.entrenar_modelo)
        btn_entrenar.pack(fill="x", pady=10)
        btn_probar = ttk.Button(frame, text="Probar modelo", command=self.probar_modelo)
        btn_probar.pack(fill="x", pady=10)
        btn_reentrenar = ttk.Button(frame, text="RE Entrenamiento", command=self.reentrenar_modelo)
        btn_reentrenar.pack(fill="x", pady=10)

    def abrir_bordes(self):
        try:
            from gui.detector_bordes_gui import DetectorBordesGUI
            self.root.destroy()
            DetectorBordesGUI()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el detector de bordes:\n{e}")

    def entrenar_modelo(self):
        try:
            from model.deep_sugarcane_model import train_model
            log_window = tk.Toplevel(self.root)
            log_window.title("Seguimiento del Entrenamiento")
            log_window.geometry("700x400")
            log_text = tk.Text(log_window, wrap="word", state="disabled", bg="#222", fg="#0f0", font=("Consolas", 10))
            log_text.pack(expand=True, fill="both")
            log_window.update()
            import threading, sys
            class ConsoleRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                def write(self, msg):
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert("end", msg)
                    self.text_widget.see("end")
                    self.text_widget.configure(state="disabled")
                def flush(self):
                    pass
            original_stdout = sys.stdout
            class Tee:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, msg):
                    for s in self.streams:
                        s.write(msg)
                def flush(self):
                    for s in self.streams:
                        s.flush()
            sys.stdout = Tee(original_stdout, ConsoleRedirector(log_text))
            def run_training():
                try:
                    train_model(train_dir="data/model_training/train", val_dir="data/model_training/val")
                    log_text.configure(state="normal")
                    log_text.insert("end", "\nEntrenamiento completado.\n")
                    log_text.configure(state="disabled")
                    messagebox.showinfo("Entrenamiento", "Entrenamiento completado.")
                except Exception as e:
                    log_text.configure(state="normal")
                    log_text.insert("end", f"\nError durante el entrenamiento:\n{e}\n")
                    log_text.configure(state="disabled")
                    messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")
                finally:
                    sys.stdout = original_stdout
            threading.Thread(target=run_training, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")

    def obtener_modelos_disponibles(self):
        modelos_dir = os.path.join(os.path.dirname(__file__), "model", )
        modelos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
        if not os.path.exists(modelos_dir):
            return []
        modelos = [f for f in os.listdir(modelos_dir) if f.endswith(".pth")]
        return modelos

    def reentrenar_modelo(self):
        try:
            from model.deep_sugarcane_model import train_model
            from tkinter import filedialog
            modelos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
            modelos = self.obtener_modelos_disponibles()
            if not modelos:
                messagebox.showwarning("Advertencia", "No se encontraron modelos disponibles para reentrenar.")
                return
            ruta_modelo = filedialog.askopenfilename(
                title="Seleccionar modelo para reentrenar",
                initialdir=modelos_dir,
                filetypes=[("Modelos PyTorch", "*.pth")]
            )
            if ruta_modelo:
                log_window = tk.Toplevel(self.root)
                log_window.title("Seguimiento del RE Entrenamiento")
                log_window.geometry("700x400")
                log_text = tk.Text(log_window, wrap="word", state="disabled", bg="#222", fg="#0f0", font=("Consolas", 10))
                log_text.pack(expand=True, fill="both")
                log_window.update()
                import threading, sys
                class ConsoleRedirector:
                    def __init__(self, text_widget):
                        self.text_widget = text_widget
                    def write(self, msg):
                        self.text_widget.configure(state="normal")
                        self.text_widget.insert("end", msg)
                        self.text_widget.see("end")
                        self.text_widget.configure(state="disabled")
                    def flush(self):
                        pass
                original_stdout = sys.stdout
                class Tee:
                    def __init__(self, *streams):
                        self.streams = streams
                    def write(self, msg):
                        for s in self.streams:
                            s.write(msg)
                    def flush(self):
                        for s in self.streams:
                            s.flush()
                sys.stdout = Tee(original_stdout, ConsoleRedirector(log_text))
                def run_retraining():
                    try:
                        train_model(train_dir="data/model_training/train", val_dir="data/model_training/val", model_path=ruta_modelo)
                        log_text.configure(state="normal")
                        log_text.insert("end", "\nRE Entrenamiento completado.\n")
                        log_text.configure(state="disabled")
                        messagebox.showinfo("RE Entrenamiento", f"Modelo seleccionado reentrenado exitosamente.")
                    except Exception as e:
                        log_text.configure(state="normal")
                        log_text.insert("end", f"\nError durante el reentrenamiento:\n{e}\n")
                        log_text.configure(state="disabled")
                        messagebox.showerror("Error", f"Error durante el reentrenamiento:\n{e}")
                    finally:
                        sys.stdout = original_stdout
                threading.Thread(target=run_retraining, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el reentrenamiento:\n{e}")

    def probar_modelo(self):
        try:
            import sys
            import importlib.util
            app_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")
            if os.path.exists(app_path):
                self.root.destroy()
                # Ejecutar la interfaz de clasificación
                import subprocess
                subprocess.Popen([sys.executable, app_path])
            else:
                messagebox.showerror("Error", "No se encontró la interfaz de prueba de modelo.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la interfaz de prueba de modelo:\n{e}")

# Si se ejecuta este archivo directamente, mostrar la interfaz principal
if __name__ == "__main__":
    MainInterface()
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🖼️ Detector de Bordes")
        # Configurar ventana a pantalla completa
        ancho_pantalla = self.root.winfo_screenwidth()
        alto_pantalla = self.root.winfo_screenheight()
        self.root.geometry(f"{ancho_pantalla}x{alto_pantalla}+0+0")
        self.root.state('zoomed')
        self.root.configure(bg="#F1F6F9")

        # Inicializar manejadores
        self.file_manager = FileManager()
        self.image_processor = ImageProcessor()

        # Variables
        self.imagen_path = None
        self.info_imagen_raw = None
        self.info_resultado = None
        self.imagen_procesada = None  # Mantener la imagen procesada original
        self.modo_grises = tk.BooleanVar(value=False)  # switch de modo grises

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

        # Configuración del estilo del switch
        estilo.configure("Switch.TCheckbutton",
                        background=fondo,
                        font=("Segoe UI", 11),
                        foreground=color_primario)
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

        # Switch de modo grises
        self.switch_grises = ttk.Checkbutton(main_frame, text="🎨 Modo grises",
                                            variable=self.modo_grises,
                                            style="Switch.TCheckbutton")
        self.switch_grises.pack(pady=10)

        # Botón para procesar
        self.btn_procesar = ttk.Button(main_frame, text="⚙️ Procesar Imagen", command=self.procesar_imagen, state='disabled')
        self.btn_procesar.pack(pady=10, fill='x')

        # Etiqueta para mostrar estado
        self.lbl_estado = ttk.Label(main_frame, text="", wraplength=560,
                                    foreground="#1E40AF", justify="center")
        self.lbl_estado.pack(pady=10)
        
        # Área para vista previa de imágenes
        self.frame_vista_previa = ttk.Frame(main_frame, padding=10)
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        self.frame_vista_previa.pack_forget()  # Inicialmente oculto

        # Frame para contener las imágenes lado a lado
        self.frame_imagenes = ttk.Frame(self.frame_vista_previa)
        self.frame_imagenes.pack(fill='both', expand=True)

        # Frame para imagen original
        self.frame_original = ttk.Frame(self.frame_imagenes)
        self.frame_original.pack(side=tk.LEFT, padx=10, fill='both', expand=True)
        ttk.Label(self.frame_original, text="Original", style="Titulo.TLabel").pack(pady=(0,10))
        self.lbl_original = ttk.Label(self.frame_original)
        self.lbl_original.pack(pady=10)

        # Frame para imagen procesada
        self.frame_procesada = ttk.Frame(self.frame_imagenes)
        self.frame_procesada.pack(side=tk.LEFT, padx=10, fill='both', expand=True)
        ttk.Label(self.frame_procesada, text="Procesada", style="Titulo.TLabel").pack(pady=(0,10))
        self.lbl_vista_previa = ttk.Label(self.frame_procesada)
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
            if not self.imagen_path or not hasattr(self, 'info_imagen_raw'):
                return  # No procesar si no hay imagen cargada

            resultado = self.image_processor.procesar_imagen_completa(
                self.imagen_path
            )
            
            # Guardar la imagen procesada original
            self.imagen_procesada = resultado['imagen_A4'].copy()
            
            # Preparar información para guardar resultados, pero no guardar aún
            self.info_resultado = self.file_manager.preparar_resultados(
                resultado['imagen_A4'],
                self.imagen_path
            )
            
            # El switch de modo grises siempre está habilitado
            self.switch_grises.config(state='active')
            self.modo_grises.trace_add('write', lambda *args: self.actualizar_vista_previa())
            
            # Mostrar vista previa en la interfaz principal
            self.mostrar_vista_previa_integrada(self.imagen_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al procesar la imagen:\n{str(e)}")

    def confirmar_guardado(self, info_imagen_raw, info_resultado):
        """Guarda tanto la imagen original como la procesada."""
        try:
            # Guardar la imagen original en raw
            nombre_raw = self.file_manager.guardar_imagen_raw(info_imagen_raw)
            
            # Si el modo grises está activo, convertir la imagen antes de guardar
            if self.modo_grises.get():
                from img_proc.esc_grises import convertir_a_grises
                imagen_a_guardar = convertir_a_grises(info_resultado['imagen'])
                info_resultado['imagen'] = imagen_a_guardar
            
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
        # Limpiar la imagen procesada y deshabilitar el switch
        self.imagen_procesada = None
        self.modo_grises.set(False)
        self.switch_grises.config(state='enable')
        
    def actualizar_vista_previa(self):
        """Actualiza la vista previa cuando cambia el modo de grises."""
        if self.imagen_procesada is not None:
            self.mostrar_vista_previa_integrada(self.imagen_procesada)

    def mostrar_vista_previa_integrada(self, imagen):
        """Muestra la vista previa de la imagen original y procesada en la interfaz principal."""
        # Mostrar imagen original
        imagen_original = cv2.imread(self.imagen_path)
        imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
        
        # Redimensionar imagen original
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
        
        # Convertir imagen original a formato tkinter
        imagen_original_pil = Image.fromarray(imagen_original_rgb)
        imagen_original_tk = ImageTk.PhotoImage(imagen_original_pil)
        self.lbl_original.config(image=imagen_original_tk)
        self.lbl_original.image = imagen_original_tk
        
        # Procesar imagen
        if isinstance(imagen, np.ndarray):
            imagen_mostrar = imagen.copy()
            
            # Aplicar filtro de escala de grises si está activado
            if self.modo_grises.get():
                from img_proc.esc_grises import convertir_a_grises
                imagen_mostrar = convertir_a_grises(imagen_mostrar)
                imagen_rgb = cv2.cvtColor(imagen_mostrar, cv2.COLOR_GRAY2RGB)
            else:
                if len(imagen_mostrar.shape) == 3 and imagen_mostrar.shape[2] == 3:
                    imagen_rgb = cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB)
                else:
                    imagen_rgb = imagen_mostrar
            
            # Redimensionar imagen procesada
            alto, ancho = imagen_rgb.shape[:2]
            if alto > max_size or ancho > max_size:
                if alto > ancho:
                    nuevo_alto = max_size
                    nuevo_ancho = int(ancho * (max_size / alto))
                else:
                    nuevo_ancho = max_size
                    nuevo_alto = int(alto * (max_size / ancho))
                imagen_rgb = cv2.resize(imagen_rgb, (nuevo_ancho, nuevo_alto))
            
            # Convertir a formato tkinter
            imagen_pil = Image.fromarray(imagen_rgb)
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
        else:
            imagen_tk = ImageTk.PhotoImage(imagen)
        
        # Mostrar imagen procesada
        self.lbl_vista_previa.config(image=imagen_tk)
        self.lbl_vista_previa.image = imagen_tk
        
        # Mostrar el frame de vista previa
        self.frame_vista_previa.pack(pady=10, fill='both', expand=True)
        
        # Actualizar estado
        self.lbl_estado.config(text="✅ Imagen procesada. Confirme para guardar.")
    
    def limpiar_estado(self):
        """Limpia el estado de la aplicación después de procesar una imagen."""
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
        """Acepta la imagen procesada y la guarda."""
        self.confirmar_guardado(self.info_imagen_raw, self.info_resultado)
        self.limpiar_estado()
    
    def rechazar_imagen(self):
        """Rechaza la imagen procesada."""
        self.cancelar_guardado()
        self.limpiar_estado()
        self.frame_vista_previa.pack_forget()  # Ocultar vista previa
    
    def iniciar(self):
        """Inicia el loop principal de la interfaz gráfica."""
        self.root.mainloop()