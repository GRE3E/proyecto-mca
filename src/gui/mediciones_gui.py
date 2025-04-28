import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from img_proc.main_processor import ImageProcessor
from img_proc.medicion_cana import medir_cana_y_nudos, ESCALA_CM_POR_PIXEL, medir_cana_y_nudos_con_escala_dinamica, ESCALA_CM_POR_PIXEL_NUEVA

class MedicionesGUI:
    def __init__(self):
        self.root = tk.Toplevel()
        self.root.title("Medición de Caña de Azúcar")
        self.root.geometry("1200x1200")
        self.root.configure(bg="#F1F6F9")
        self.resultados = None
        self._build_ui()
        self.root.mainloop()

    def _build_ui(self):
        # Título principal
        titulo = tk.Label(self.root, text="Medición de Caña de Azúcar", font=("Segoe UI", 24, "bold"), bg="#F1F6F9", fg="#1E40AF")
        titulo.pack(pady=(20, 30))

        # Contenedor principal de tres columnas
        main_container = tk.Frame(self.root, bg="#F1F6F9")
        main_container.pack(expand=True, fill="both", padx=20)

        # Columna 1: Imagen Original
        col1 = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief="solid")
        col1.pack(side="left", fill="both", expand=True, padx=10)
        tk.Label(col1, text="Imagen Original", font=("Segoe UI", 14, "bold"), bg="#FFFFFF", fg="#1E40AF").pack(pady=10)
        self.original_label = tk.Label(col1, bg="#FFFFFF")
        self.original_label.pack(pady=10, padx=10)

        # Columna 2: Imagen Procesada
        col2 = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief="solid")
        col2.pack(side="left", fill="both", expand=True, padx=10)
        tk.Label(col2, text="Imagen Procesada", font=("Segoe UI", 14, "bold"), bg="#FFFFFF", fg="#1E40AF").pack(pady=10)
        self.image_label = tk.Label(col2, bg="#FFFFFF")
        self.image_label.pack(pady=10, padx=10)

        # Columna 3: Resultados de Medición con Scrollbar
        col3 = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief="solid")
        col3.pack(side="left", fill="both", expand=True, padx=10)
        tk.Label(col3, text="Resultados de Medición", font=("Segoe UI", 14, "bold"), bg="#FFFFFF", fg="#1E40AF").pack(pady=10)

        # Crear un Canvas con Scrollbar
        canvas = tk.Canvas(col3, bg="#FFFFFF")
        scrollbar = tk.Scrollbar(col3, orient="vertical", command=canvas.yview)
        self.panel_resultados = tk.Frame(canvas, bg="#FFFFFF")

        # Configurar el Canvas y el Scrollbar
        self.panel_resultados.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.panel_resultados, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Empaquetar el Canvas y el Scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Panel de configuración
        panel_config = tk.Frame(self.root, bg="#F1F6F9")
        panel_config.pack(fill="x", pady=10, padx=20)
        
        # Información de escala actualizada
        tk.Label(panel_config, text=f"Escala: {ESCALA_CM_POR_PIXEL_NUEVA:.4f} cm/píxel", 
                font=("Segoe UI", 12), bg="#F1F6F9", fg="#1E40AF").pack(side="left", padx=5)

        # Panel de botones
        panel_botones = tk.Frame(self.root, bg="#F1F6F9")
        panel_botones.pack(fill="x", pady=20, padx=20)

        # Botón Cargar Imagen
        btn_cargar = tk.Button(panel_botones, text="Cargar Imagen", command=self.select_image,
                             font=("Segoe UI", 12, "bold"), bg="#1E40AF", fg="#FFF", width=15)
        btn_cargar.pack(side="left", padx=5)

        # Botón Medir Caña
        btn_medir = tk.Button(panel_botones, text="Medir Caña", command=self.medir_cana,
                            font=("Segoe UI", 12, "bold"), bg="#22C55E", fg="#FFF", width=15)
        btn_medir.pack(side="left", padx=5)

        # Botón Guardar Resultados
        btn_guardar = tk.Button(panel_botones, text="Guardar Resultados", command=self.guardar_resultados,
                              font=("Segoe UI", 12, "bold"), bg="#F59E42", fg="#FFF", width=15)
        btn_guardar.pack(side="left", padx=5)

        # Botón Volver al Menú Principal
        btn_volver = tk.Button(panel_botones, text="Volver al Menú Principal", command=self.root.destroy,
                             font=("Segoe UI", 12, "bold"), bg="#64748B", fg="#FFF", width=18)
        btn_volver.pack(side="right", padx=5)

        # Mensaje inicial
        self._mostrar_mensaje("Seleccione una imagen para comenzar el análisis")

    def _mostrar_mensaje(self, mensaje):
        for widget in self.panel_resultados.winfo_children():
            widget.destroy()
        label = tk.Label(self.panel_resultados, text=mensaje, font=("Segoe UI", 12), bg="#FFFFFF", fg="#64748B", wraplength=300)
        label.pack(pady=20)

    def select_image(self):
        filetypes = [("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        img_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
        if img_path:
            try:
                # Mostrar imagen original
                img_original = cv2.imread(img_path)
                img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                pil_img_original = Image.fromarray(img_original)
                pil_img_original.thumbnail((400, 400))
                img_tk_original = ImageTk.PhotoImage(pil_img_original)
                self.original_label.configure(image=img_tk_original)
                self.original_label.image = img_tk_original

                # Guardar la ruta para el procesamiento
                self.imagen_path = img_path
                self._mostrar_mensaje("Imagen cargada. Presione 'Medir Caña' para procesar.")
            except Exception as e:
                self._mostrar_mensaje(f"Error al cargar la imagen: {str(e)}")
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def medir_cana(self):
        if not hasattr(self, 'imagen_path'):
            messagebox.showwarning("Advertencia", "Por favor, primero seleccione una imagen.")
            return

        try:
            processor = ImageProcessor()
            processed_result = processor.procesar_imagen_completa(self.imagen_path)
            
            # Procesar imagen
            imagen_a4 = processed_result.get('imagen_A4', None)
            if imagen_a4 is None:
                imagen_a4 = cv2.imread(self.imagen_path)
            
            # Usar la nueva función con escala dinámica
            resultados = medir_cana_y_nudos_con_escala_dinamica(imagen_a4)
            self.resultados = resultados

            # Mostrar imagen procesada con resultados visualizados
            if 'imagen_resultado' in resultados:
                processed_img = resultados['imagen_resultado']
            else:
                processed_img = processed_result['imagen_bordes']

            # Mostrar imagen procesada
            if isinstance(processed_img, np.ndarray):
                if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(processed_img)
            else:
                pil_img = processed_img

            pil_img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(pil_img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

            # Mostrar resultados
            self._mostrar_resultados_detallados(resultados)

        except Exception as e:
            self._mostrar_mensaje(f"Error al procesar la imagen: {str(e)}")
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{e}")

    def _mostrar_resultados_detallados(self, resultados):
        for widget in self.panel_resultados.winfo_children():
            widget.destroy()

        # Contenedor para los resultados
        frame_resultados = tk.Frame(self.panel_resultados, bg="#FFFFFF")
        frame_resultados.pack(fill="both", expand=True)

        # Medidas principales
        tk.Label(frame_resultados, text="Medidas Principales", font=("Segoe UI", 12, "bold"), 
                bg="#FFFFFF", fg="#1E40AF").pack(pady=(0,10))

        # Crear marco para las medidas
        frame_medidas = tk.Frame(frame_resultados, bg="#FFFFFF")
        frame_medidas.pack(fill="x", padx=10)

        # Mostrar medidas principales
        medidas = [
            ("Largo total:", f"{resultados['largo_cana_cm']:.2f} cm"),
            ("Largo total (px):", f"{resultados['largo_cana_px']} px"),
            ("Ancho total:", f"{resultados['ancho_cana_cm']:.2f} cm"),
            ("Ancho total (px):", f"{resultados['ancho_cana_px']} px"),
            ("Cantidad de nudos:", str(resultados['cantidad_nudos'])),
            ("Cantidad de entrenudos:", str(resultados['cantidad_entrenudos']))
        ]

        for i, (label, valor) in enumerate(medidas):
            tk.Label(frame_medidas, text=label, font=("Segoe UI", 10), bg="#FFFFFF", fg="#64748B").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(frame_medidas, text=valor, font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E40AF").grid(row=i, column=1, sticky="w", padx=10, pady=2)

        # Detalles de Nudos
        tk.Label(frame_resultados, text="Detalles de Nudos", font=("Segoe UI", 12, "bold"), 
                bg="#FFFFFF", fg="#22C55E").pack(pady=(20,10))

        # Marco para los detalles de nudos
        frame_nudos = tk.Frame(frame_resultados, bg="#FFFFFF")
        frame_nudos.pack(fill="x", padx=10)

        for i, n in enumerate(resultados['nudos']):
            texto = f"Nudo {i+1}: Largo {n['largo_cm']:.2f} cm, Ancho {n['ancho_cm']:.2f} cm"
            tk.Label(frame_nudos, text=texto, font=("Segoe UI", 10), 
                    bg="#FFFFFF", fg="#64748B").pack(anchor="w", pady=2)

        # Detalles de Entrenudos
        tk.Label(frame_resultados, text="Detalles de Entrenudos", font=("Segoe UI", 12, "bold"), 
                bg="#FFFFFF", fg="#F59E42").pack(pady=(20,10))

        # Marco para los detalles de entrenudos
        frame_entrenudos = tk.Frame(frame_resultados, bg="#FFFFFF")
        frame_entrenudos.pack(fill="x", padx=10)

        for i, e in enumerate(resultados['entrenudos']):
            texto = f"Entrenudo {i+1}: Largo {e['largo_cm']:.2f} cm, Ancho {e['ancho_cm']:.2f} cm"
            tk.Label(frame_entrenudos, text=texto, font=("Segoe UI", 10), 
                    bg="#FFFFFF", fg="#64748B").pack(anchor="w", pady=2)

    def guardar_resultados(self):
        if not hasattr(self, 'resultados') or not self.resultados:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar.")
            return
        
        try:
            # Guardar imagen procesada
            filetypes = [("Imágenes PNG", "*.png"), ("Imágenes JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
            img_path = filedialog.asksaveasfilename(title="Guardar imagen procesada", 
                                                  defaultextension=".png", filetypes=filetypes)
            if img_path:
                if 'imagen_resultado' in self.resultados:
                    cv2.imwrite(img_path, self.resultados['imagen_resultado'])
                else:
                    messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")
            
            # Guardar resultados en texto
            filetypes = [("Archivos de texto", "*.txt"), ("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
            txt_path = filedialog.asksaveasfilename(title="Guardar resultados en texto", 
                                                  defaultextension=".txt", filetypes=filetypes)
            if txt_path:
                with open(txt_path, 'w') as f:
                    f.write(self._generar_texto_resultados(self.resultados))
                
            if img_path or txt_path:
                messagebox.showinfo("Éxito", "Resultados guardados correctamente.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar los resultados:\n{e}")

    def _generar_texto_resultados(self, resultados):
        texto = f"RESULTADOS DE MEDICIÓN DE CAÑA DE AZÚCAR\n"
        texto += f"=====================================\n\n"
        texto += f"Largo caña (px): {resultados['largo_cana_px']}\n"
        texto += f"Largo caña (cm): {resultados['largo_cana_cm']:.2f}\n"
        texto += f"Ancho caña (px): {resultados['ancho_cana_px']}\n"
        texto += f"Ancho caña (cm): {resultados['ancho_cana_cm']:.2f}\n"
        texto += f"Cantidad Nudos: {resultados['cantidad_nudos']}\n"
        texto += f"Cantidad Entrenudos: {resultados['cantidad_entrenudos']}\n\n"
        
        texto += "Nudos:\n"
        for i, n in enumerate(resultados['nudos']):
            texto += f"  Nudo {i+1}: Largo(px): {n['largo_px']}, Ancho(px): {n['ancho_px']}"
            texto += f", Largo(cm): {n['largo_cm']:.2f}, Ancho(cm): {n['ancho_cm']:.2f}\n"
        
        texto += "\nEntrenudos:\n"
        for i, e in enumerate(resultados['entrenudos']):
            texto += f"  Entrenudo {i+1}: Largo(px): {e['largo_px']}, Ancho(px): {e['ancho_px']}"
            texto += f", Largo(cm): {e['largo_cm']:.2f}, Ancho(cm): {e['ancho_cm']:.2f}\n"
        
        return texto