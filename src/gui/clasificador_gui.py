import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys

# Importar módulos necesarios
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.deep_sugarcane_model import predict_image
from img_proc.medicion_cana import medir_cana_y_nudos_con_escala_dinamica

class ClasificadorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clasificador de Caña de Azúcar")
        self.root.geometry("1200x800")
        self.root.configure(bg="#F1F6F9")
        
        # Variables
        self.imagen_path = None
        self.relacion_pixel_cm = tk.DoubleVar(value=10.0)
        self.resultados_modelo = None
        
        # Construir la interfaz
        self._build_ui()
        
        self.root.mainloop()
    
    def _build_ui(self):
        # Título principal
        titulo = tk.Label(self.root, text="Clasificador de Caña de Azúcar", 
                         font=("Segoe UI", 24, "bold"), bg="#F1F6F9", fg="#1E40AF")
        titulo.pack(pady=(20, 30))
        
        # Panel de configuración
        panel_config = tk.Frame(self.root, bg="#F1F6F9", bd=1, relief="solid")
        panel_config.pack(fill="x", padx=20, pady=10)
        
        # Relación Píxel-Centímetro
        tk.Label(panel_config, text="Relación Píxel-Centímetro:", 
                font=("Segoe UI", 12), bg="#F1F6F9").pack(side="left", padx=10, pady=10)
        
        entry_relacion = tk.Entry(panel_config, textvariable=self.relacion_pixel_cm, 
                                 width=10, font=("Segoe UI", 12))
        entry_relacion.pack(side="left", padx=5, pady=10)
        
        tk.Label(panel_config, text="píxeles/cm", 
                font=("Segoe UI", 12), bg="#F1F6F9").pack(side="left", padx=5, pady=10)
        
        btn_actualizar = tk.Button(panel_config, text="Actualizar", 
                                  command=self.actualizar_relacion,
                                  font=("Segoe UI", 12), bg="#3B82F6", fg="white")
        btn_actualizar.pack(side="left", padx=20, pady=10)
        
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Pestaña Imagen
        self.tab_imagen = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_imagen, text="Imagen")
        
        # Pestaña Resultados
        self.tab_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_resultados, text="Resultados")
        
        # Contenido de la pestaña Imagen
        self.frame_imagen = tk.Frame(self.tab_imagen, bg="white")
        self.frame_imagen.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_instruccion = tk.Label(self.frame_imagen, 
                                      text="Seleccione una imagen para predicción",
                                      font=("Segoe UI", 14), bg="white", fg="#1E40AF")
        self.lbl_instruccion.pack(pady=20)
        
        self.lbl_confianza = tk.Label(self.frame_imagen, 
                                    text="Confianza: N/A",
                                    font=("Segoe UI", 12), bg="white", fg="#64748B")
        self.lbl_confianza.pack(pady=10)
        
        self.lbl_imagen = tk.Label(self.frame_imagen, bg="white")
        self.lbl_imagen.pack(pady=20, expand=True)
        
        # Contenido de la pestaña Resultados
        self.frame_resultados = tk.Frame(self.tab_resultados, bg="white")
        self.frame_resultados.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Crear un Canvas con Scrollbar para los resultados
        self.canvas_resultados = tk.Canvas(self.frame_resultados, bg="white")
        scrollbar = ttk.Scrollbar(self.frame_resultados, orient="vertical", 
                                 command=self.canvas_resultados.yview)
        
        self.panel_resultados = tk.Frame(self.canvas_resultados, bg="white")
        self.panel_resultados.bind(
            "<Configure>",
            lambda e: self.canvas_resultados.configure(scrollregion=self.canvas_resultados.bbox("all"))
        )
        
        self.canvas_resultados.create_window((0, 0), window=self.panel_resultados, anchor="nw")
        self.canvas_resultados.configure(yscrollcommand=scrollbar.set)
        
        self.canvas_resultados.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Panel de botones
        panel_botones = tk.Frame(self.root, bg="#F1F6F9")
        panel_botones.pack(fill="x", pady=20, padx=20)
        
        # Botón Seleccionar imagen
        btn_seleccionar = tk.Button(panel_botones, text="Seleccionar imagen", 
                                   command=self.seleccionar_imagen,
                                   font=("Segoe UI", 12, "bold"), bg="#1E40AF", fg="white", 
                                   width=18)
        btn_seleccionar.pack(side="left", padx=5)
        
        # Botón Probar modelo
        self.btn_probar = tk.Button(panel_botones, text="Probar modelo", 
                                  command=self.probar_modelo,
                                  font=("Segoe UI", 12, "bold"), bg="#22C55E", fg="white", 
                                  width=15, state="disabled")
        self.btn_probar.pack(side="left", padx=5)
        
        # Botón Volver al menú principal
        btn_volver = tk.Button(panel_botones, text="Volver al menú principal", 
                              command=self.volver_menu,
                              font=("Segoe UI", 12, "bold"), bg="#64748B", fg="white", 
                              width=18)
        btn_volver.pack(side="right", padx=5)
    
    def seleccionar_imagen(self):
        """Permite al usuario seleccionar una imagen para análisis"""
        filetypes = [("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        img_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
        
        if img_path:
            try:
                self.imagen_path = img_path
                self.mostrar_imagen(img_path)
                self.btn_probar.config(state="normal")
                self.lbl_instruccion.config(text="Imagen cargada. Presione 'Probar modelo' para analizar.")
                
                # Automáticamente probar el modelo al cargar la imagen
                self.probar_modelo()
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
    
    def mostrar_imagen(self, img_path):
        """Muestra la imagen seleccionada en la interfaz"""
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para mostrar
            alto, ancho = img_rgb.shape[:2]
            max_size = 600
            
            if alto > ancho:
                nuevo_alto = max_size
                nuevo_ancho = int(ancho * (max_size / alto))
            else:
                nuevo_ancho = max_size
                nuevo_alto = int(alto * (max_size / ancho))
            
            img_resized = cv2.resize(img_rgb, (nuevo_ancho, nuevo_alto))
            
            # Convertir a formato para tkinter
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Mostrar en la interfaz
            self.lbl_imagen.config(image=img_tk)
            self.lbl_imagen.image = img_tk  # Mantener referencia
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar la imagen:\n{str(e)}")
    
    def actualizar_relacion(self):
        """Actualiza la relación píxel-centímetro"""
        try:
            # Si hay resultados previos, actualizar con la nueva relación
            if self.resultados_modelo:
                self.probar_modelo()
            messagebox.showinfo("Información", "Relación píxel-centímetro actualizada.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar la relación:\n{str(e)}")
    
    def probar_modelo(self):
        """Procesa la imagen con el modelo y muestra los resultados"""
        if not self.imagen_path:
            messagebox.showwarning("Advertencia", "Por favor, primero seleccione una imagen.")
            return
        
        try:
            # Obtener la relación píxel-cm
            pixel_to_cm_ratio = self.relacion_pixel_cm.get()
            
            # Predecir con el modelo
            from model.deep_sugarcane_model import predict_image
            resultados = predict_image(self.imagen_path)
            self.resultados_modelo = resultados
            
            # Mostrar la imagen con las mediciones visualizadas
            if 'imagen_resultado' in resultados:
                img_resultado = resultados['imagen_resultado']
                img_pil = Image.fromarray(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
                img_pil.thumbnail((600, 600))
                img_tk = ImageTk.PhotoImage(img_pil)
                self.lbl_imagen.config(image=img_tk)
                self.lbl_imagen.image = img_tk
            
            # Mostrar resultados detallados
            self.mostrar_resultados_detallados(resultados)
            
            # Cambiar a la pestaña de resultados
            self.notebook.select(1)  # Seleccionar la pestaña de resultados
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def mostrar_resultados_detallados(self, resultados):
        """Muestra los resultados detallados en la pestaña de resultados"""
        # Limpiar resultados anteriores
        for widget in self.panel_resultados.winfo_children():
            widget.destroy()
        
        # Verificar si es una caña
        if 'Caña' in resultados and resultados['Caña'] is not False:
            # Título
            tk.Label(self.panel_resultados, text="Resultados del Análisis", 
                    font=("Segoe UI", 16, "bold"), bg="white", fg="#1E40AF").pack(pady=(10, 20))
            
            # Medidas principales
            frame_medidas = tk.Frame(self.panel_resultados, bg="white")
            frame_medidas.pack(fill="x", padx=20, pady=10)
            
            tk.Label(frame_medidas, text="Medidas Principales", 
                    font=("Segoe UI", 14, "bold"), bg="white", fg="#1E40AF").pack(anchor="w")
            
            # Altura
            if 'AltoRecto_cm' in resultados['Caña']:
                alto_recto = resultados['Caña']['AltoRecto_cm']
                tk.Label(frame_medidas, text=f"Altura recta: {alto_recto:.2f} cm", 
                        font=("Segoe UI", 12), bg="white").pack(anchor="w", pady=2)
            
            if 'AltoCurvo_cm' in resultados['Caña']:
                alto_curvo = resultados['Caña']['AltoCurvo_cm']
                tk.Label(frame_medidas, text=f"Altura curva: {alto_curvo:.2f} cm", 
                        font=("Segoe UI", 12), bg="white").pack(anchor="w", pady=2)
            
            # Confianza
            if 'Confianza' in resultados['Caña']:
                confianza = resultados['Caña']['Confianza'] * 100
                tk.Label(frame_medidas, text=f"Confianza: {confianza:.2f}%", 
                        font=("Segoe UI", 12), bg="white").pack(anchor="w", pady=2)
            
            # Mediciones de nudos y entrenudos
            if 'Mediciones' in resultados and resultados['Mediciones']:
                # Título de mediciones
                tk.Label(self.panel_resultados, text="Mediciones de Nudos y Entrenudos", 
                        font=("Segoe UI", 14, "bold"), bg="white", fg="#22C55E").pack(pady=(20, 10))
                
                # Tabla de mediciones
                frame_tabla = tk.Frame(self.panel_resultados, bg="white")
                frame_tabla.pack(fill="x", padx=20, pady=10)
                
                # Encabezados
                tk.Label(frame_tabla, text="Nº", width=5, font=("Segoe UI", 12, "bold"), 
                        bg="#E5E7EB").grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
                tk.Label(frame_tabla, text="Nudo Largo (cm)", width=15, font=("Segoe UI", 12, "bold"), 
                        bg="#E5E7EB").grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
                tk.Label(frame_tabla, text="Nudo Ancho (cm)", width=15, font=("Segoe UI", 12, "bold"), 
                        bg="#E5E7EB").grid(row=0, column=2, padx=2, pady=2, sticky="nsew")
                tk.Label(frame_tabla, text="Entrenudo Largo (cm)", width=20, font=("Segoe UI", 12, "bold"), 
                        bg="#E5E7EB").grid(row=0, column=3, padx=2, pady=2, sticky="nsew")
                tk.Label(frame_tabla, text="Entrenudo Ancho (cm)", width=20, font=("Segoe UI", 12, "bold"), 
                        bg="#E5E7EB").grid(row=0, column=4, padx=2, pady=2, sticky="nsew")
                
                # Datos
                for i, medicion in enumerate(resultados['Mediciones']):
                    bg_color = "#F9FAFB" if i % 2 == 0 else "white"
                    
                    tk.Label(frame_tabla, text=f"{i+1}", font=("Segoe UI", 12), 
                            bg=bg_color).grid(row=i+1, column=0, padx=2, pady=2, sticky="nsew")
                    
                    tk.Label(frame_tabla, text=f"{medicion['Nudo_Largo_cm']:.2f}", font=("Segoe UI", 12), 
                            bg=bg_color).grid(row=i+1, column=1, padx=2, pady=2, sticky="nsew")
                    
                    tk.Label(frame_tabla, text=f"{medicion['Nudo_Ancho_cm']:.2f}", font=("Segoe UI", 12), 
                            bg=bg_color).grid(row=i+1, column=2, padx=2, pady=2, sticky="nsew")
                    
                    tk.Label(frame_tabla, text=f"{medicion['Entrenudo_Largo_cm']:.2f}", font=("Segoe UI", 12), 
                            bg=bg_color).grid(row=i+1, column=3, padx=2, pady=2, sticky="nsew")
                    
                    tk.Label(frame_tabla, text=f"{medicion['Entrenudo_Ancho_cm']:.2f}", font=("Segoe UI", 12), 
                            bg=bg_color).grid(row=i+1, column=4, padx=2, pady=2, sticky="nsew")
        else:
            # No es una caña
            tk.Label(self.panel_resultados, text="La imagen no contiene una caña de azúcar", 
                    font=("Segoe UI", 16, "bold"), bg="white", fg="#EF4444").pack(pady=20)
            
            if 'Confianza' in resultados['Caña']:
                confianza = resultados['Caña']['Confianza'] * 100
                tk.Label(self.panel_resultados, text=f"Confianza: {confianza:.2f}%", 
                        font=("Segoe UI", 12), bg="white").pack(pady=10)
    
    def volver_menu(self):
        """Cierra la ventana actual y vuelve al menú principal"""
        self.root.destroy()
        # Aquí puedes agregar código para volver al menú principal si es necesario

# Para ejecutar la aplicación directamente
if __name__ == "__main__":
    app = ClasificadorGUI()