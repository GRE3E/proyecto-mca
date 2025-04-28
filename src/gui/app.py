import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import json
import yaml
import math
# Añadir el directorio principal al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.deep_sugarcane_model import predict_image

def get_class_names():
    """Obtiene los nombres de las clases del archivo dataset.yaml"""
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset.yaml'))
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data['names']
    except Exception as e:
        print(f"Error al cargar dataset.yaml: {e}")
        return ["no_cana", "cana"]  # Valores por defecto

class ModelTestApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clasificador de Caña de Azúcar")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f2f5")
        self.class_names = get_class_names()
        self.pixel_to_cm_ratio = 10.0  # Valor predeterminado: 10 píxeles = 1 cm
        self.processed_img_path = None
        self.original_img_path = None
        self.prediction_result = None
        self.setup_ui()
        self.root.mainloop()

    def setup_styles(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#1E40AF", background="#f0f2f5")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 14, "bold"), foreground="#2563EB", background="#f0f2f5")
        style.configure("Normal.TLabel", font=("Segoe UI", 12), background="#f0f2f5")
        style.configure("Result.TLabel", font=("Segoe UI", 12), background="#f0f2f5", foreground="#047857")
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#FFFFFF", background="#1E40AF")
        style.map("Accent.TButton", background=[("active", "#3B82F6")])
        style.configure("Secondary.TButton", font=("Segoe UI", 11), padding=8, foreground="#1E40AF", background="#f0f2f5")
        style.map("Secondary.TButton", background=[("active", "#DBEAFE")])
        style.configure("TFrame", background="#f0f2f5")
        style.configure("Card.TFrame", background="#FFFFFF", relief="solid", borderwidth=1)
        style.configure("TNotebook", background="#f0f2f5", tabposition="n")
        style.configure("TNotebook.Tab", font=("Segoe UI", 12), padding=[12, 4], background="#FFFFFF")
        style.map("TNotebook.Tab", background=[("selected", "#1E40AF")], foreground=[("selected", "#FFFFFF")])

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Estilos
        self.setup_styles()
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_label = ttk.Label(title_frame, text="Clasificador de Caña de Azúcar", style="Title.TLabel")
        title_label.pack()
        
        # Frame para configuración de ratio píxel-cm
        ratio_frame = ttk.Frame(main_frame, padding=10, style="Card.TFrame")
        ratio_frame.pack(fill=tk.X, pady=10)
        
        ratio_label = ttk.Label(ratio_frame, text="Relación Píxel-Centímetro:", style="Normal.TLabel")
        ratio_label.pack(side=tk.LEFT, padx=10)
        
        self.ratio_var = tk.StringVar(value="10.0")
        ratio_entry = ttk.Entry(ratio_frame, textvariable=self.ratio_var, width=10)
        ratio_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ratio_frame, text="píxeles/cm", style="Normal.TLabel").pack(side=tk.LEFT)
        
        update_ratio_btn = ttk.Button(ratio_frame, text="Actualizar", 
                                     command=self.update_ratio, style="Secondary.TButton")
        update_ratio_btn.pack(side=tk.LEFT, padx=10)
        
        # Contenedor para imagen y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de imagen
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="Imagen")
        
        # Contenedor para la imagen
        image_container = ttk.Frame(self.image_tab, style="Card.TFrame", padding=15)
        image_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        self.image_label = ttk.Label(image_container)
        self.image_label.pack(expand=True)
        
        # Pestaña de resultados
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Resultados")
        
        # Contenedor para los resultados
        results_container = ttk.Frame(self.results_tab, style="Card.TFrame", padding=15)
        results_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Etiqueta para la clase predicha
        self.class_label = ttk.Label(results_container, text="Seleccione una imagen para predicción", 
                                   style="Subtitle.TLabel")
        self.class_label.pack(pady=10, anchor=tk.W)
        
        # Frame para confianza
        conf_frame = ttk.Frame(results_container)
        conf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(conf_frame, text="Confianza:", style="Normal.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        self.confidence_label = ttk.Label(conf_frame, text="N/A", style="Result.TLabel")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Frame para altura
        self.height_frame = ttk.Frame(results_container)
        ttk.Label(self.height_frame, text="Altura:", style="Normal.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        self.height_label = ttk.Label(self.height_frame, text="N/A", style="Result.TLabel")
        self.height_label.pack(side=tk.LEFT)
        
        # Frame para mediciones
        self.measurements_frame = ttk.Frame(results_container)
        self.measurements_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Título de mediciones
        self.measurements_title = ttk.Label(self.measurements_frame, 
                                          text="Mediciones", style="Subtitle.TLabel")
        
        # Treeview para mostrar las mediciones
        self.tree_columns = ("nudo_largo", "nudo_ancho", "entrenudo_largo", "entrenudo_ancho")
        self.tree = ttk.Treeview(self.measurements_frame, columns=self.tree_columns, show="headings")
        
        # Configurar columnas
        self.tree.heading("nudo_largo", text="Nudo Largo (cm)")
        self.tree.heading("nudo_ancho", text="Nudo Ancho (cm)")
        self.tree.heading("entrenudo_largo", text="Entrenudo Largo (cm)")
        self.tree.heading("entrenudo_ancho", text="Entrenudo Ancho (cm)")
        
        for col in self.tree_columns:
            self.tree.column(col, width=120, anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.measurements_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        btn_select = ttk.Button(button_frame, text="Seleccionar imagen", 
                              command=self.select_image, style="Accent.TButton")
        btn_select.pack(side=tk.LEFT, padx=10)
        
        btn_back = ttk.Button(button_frame, text="Volver al menú principal", 
                            command=self.return_to_main, style="Secondary.TButton")
        btn_back.pack(side=tk.RIGHT, padx=10)

    def update_ratio(self):
        """Actualiza la relación píxeles/cm y recalcula si hay una predicción"""
        try:
            new_ratio = float(self.ratio_var.get())
            if new_ratio <= 0:
                raise ValueError("La relación debe ser un número positivo")
            
            self.pixel_to_cm_ratio = new_ratio
            messagebox.showinfo("Actualización", f"Relación actualizada a {new_ratio} píxeles/cm")
            
            # Si ya existe una predicción, actualizar los resultados
            if self.prediction_result and self.processed_img_path:
                self.display_prediction_results()
                
        except ValueError as e:
            messagebox.showerror("Error", f"Valor inválido: {e}")

    def select_image(self):
        """Selecciona una imagen para clasificar"""
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png"),
            ("Todos los archivos", "*.*")
        ]
        img_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
        if img_path:
            self.original_img_path = img_path
            self.process_and_predict_image(img_path)

    def process_and_predict_image(self, img_path):
        """Procesa la imagen y hace la predicción"""
        try:
            # Importar el procesador de imágenes
            from img_proc.main_processor import ImageProcessor
            
            # Procesar la imagen para reducir bordes
            processor = ImageProcessor()
            processed_result = processor.procesar_imagen_completa(img_path)
            processed_img = processed_result['imagen_A4']
            
            # Guardar temporalmente la imagen procesada
            temp_dir = os.path.dirname(img_path)
            self.processed_img_path = os.path.join(temp_dir, "temp_processed.jpg")
            
            # Convertir la imagen procesada a formato PIL y guardarla
            if isinstance(processed_img, np.ndarray):
                if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(processed_img)
                pil_img.save(self.processed_img_path)
            else:
                # Si ya es un objeto PIL Image
                processed_img.save(self.processed_img_path)
            
            # Mostrar la imagen en la interfaz
            self.display_image(self.processed_img_path)
            
            # Enviar la imagen procesada al modelo para predicción
            self.prediction_result = predict_image(self.processed_img_path)
            
            # Mostrar los resultados
            self.display_prediction_results()
            
            # Cambiar a la pestaña de resultados
            self.notebook.select(self.results_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_tb(e.__traceback__)

    def display_image(self, img_path):
        """Muestra la imagen en la interfaz"""
        try:
            # Cargar la imagen
            pil_img = Image.open(img_path)
            
            # Redimensionar para mostrar en la interfaz
            width, height = pil_img.size
            max_size = 500
            
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
                
            resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(resized_img)
            
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # Mantener una referencia
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen:\n{e}")

    def display_prediction_results(self):
        """Muestra los resultados de la predicción"""
        if not self.prediction_result:
            return
            
        # Limpiar resultados anteriores
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        pred = self.prediction_result
        
        # Verificar si es una caña
        if isinstance(pred, dict) and "Caña" in pred:
            cane_data = pred["Caña"]
            
            if isinstance(cane_data, dict):  # Es una caña
                # Mostrar la clase
                self.class_label.config(text=f"Predicción: {self.class_names[1]}")
                
                # Mostrar confianza
                confidence = cane_data.get("Confianza", 0) * 100
                self.confidence_label.config(text=f"{confidence:.2f}%")
                
                # Mostrar altura aplicando la escala
                alto_recto = cane_data.get("AltoRecto_cm", 0)
                alto_curvo = cane_data.get("AltoCurvo_cm", 0)
                
                # Mostrar el frame de altura
                self.height_frame.pack(fill=tk.X, pady=5)
                self.height_label.config(text=f"Recto: {alto_recto:.2f} cm, Curvo: {alto_curvo:.2f} cm")
                
                # Mostrar mediciones
                self.measurements_title.pack(pady=(15, 5), anchor=tk.W)
                self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
                
                # Procesar mediciones
                measurements = pred.get("Mediciones", [])
                if measurements:
                    for i, m in enumerate(measurements):
                        if m:  # Verificar que no sean valores nulos
                            # Aplicar la escala del modelo a centímetros
                            nudo_largo = float(m.get("Nudo_Largo_cm", 0))
                            nudo_ancho = float(m.get("Nudo_Ancho_cm", 0))
                            entrenudo_largo = float(m.get("Entrenudo_Largo_cm", 0))
                            entrenudo_ancho = float(m.get("Entrenudo_Ancho_cm", 0))
                            
                            # Si alguno es mayor que cero, agregar al árbol
                            if nudo_largo > 0 or nudo_ancho > 0 or entrenudo_largo > 0 or entrenudo_ancho > 0:
                                self.tree.insert("", tk.END, values=(
                                    f"{nudo_largo:.2f}", 
                                    f"{nudo_ancho:.2f}", 
                                    f"{entrenudo_largo:.2f}", 
                                    f"{entrenudo_ancho:.2f}"
                                ))
            else:  # No es una caña
                self.class_label.config(text=f"Predicción: {self.class_names[0]}")
                self.confidence_label.config(text=f"{(1 - cane_data) * 100:.2f}%")
                
                # Ocultar frames de altura y mediciones
                self.height_frame.pack_forget()
                self.measurements_title.pack_forget()
                self.tree.pack_forget()
        else:
            # Formato desconocido de predicción
            self.class_label.config(text="Predicción desconocida")
            self.confidence_label.config(text="N/A")
            self.height_frame.pack_forget()
            self.measurements_title.pack_forget()
            self.tree.pack_forget()

    def return_to_main(self):
        """Vuelve al menú principal"""
        self.root.destroy()
        try:
            import sys
            import os
            import subprocess
            app_main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app_main.py'))
            subprocess.Popen([sys.executable, app_main_path])
        except Exception as e:
            print(f"Error al volver al menú principal: {e}")

    def cleanup(self):
        """Limpia archivos temporales"""
        if self.processed_img_path and os.path.exists(self.processed_img_path):
            try:
                os.remove(self.processed_img_path)
            except:
                pass

if __name__ == "__main__":
    app = ModelTestApp()
    app.cleanup()