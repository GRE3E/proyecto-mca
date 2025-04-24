import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from img_proc.main_processor import ImageProcessor

class MedicionesGUI:
    def __init__(self):
        self.root = tk.Toplevel()
        self.root.title("Medición de Caña de Azúcar")
        self.root.geometry("1200x800")
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

        # Columna 3: Resultados de Medición
        col3 = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief="solid")
        col3.pack(side="left", fill="both", expand=True, padx=10)
        tk.Label(col3, text="Resultados de Medición", font=("Segoe UI", 14, "bold"), bg="#FFFFFF", fg="#1E40AF").pack(pady=10)
        self.panel_resultados = tk.Frame(col3, bg="#FFFFFF")
        self.panel_resultados.pack(fill="both", expand=True, padx=10, pady=10)

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
            processed_img = processed_result['imagen_bordes']
            
            # Procesar imagen
            from img_proc.medicion_cana import medir_cana_y_nudos
            imagen_a4 = processed_result.get('imagen_A4', None)
            if imagen_a4 is None:
                imagen_a4 = cv2.imread(self.imagen_path)
            resultados = medir_cana_y_nudos(imagen_a4)
            self.resultados = resultados

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
            ("Largo total:", f"{resultados['alto_cana_px']} px"),
            ("Ancho total:", "4 cm"),
            ("Cantidad de nudos:", str(resultados['cantidad_nudos'])),
            ("Cantidad de entrenudos:", str(resultados['cantidad_entrenudos']))
        ]

        for i, (label, valor) in enumerate(medidas):
            tk.Label(frame_medidas, text=label, font=("Segoe UI", 10), bg="#FFFFFF", fg="#64748B").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(frame_medidas, text=valor, font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1E40AF").grid(row=i, column=1, sticky="w", padx=10, pady=2)

        # Detalles de Entrenudos
        tk.Label(frame_resultados, text="Detalles de Entrenudos", font=("Segoe UI", 12, "bold"), 
                bg="#FFFFFF", fg="#F59E42").pack(pady=(20,10))

        # Marco para los detalles de entrenudos
        frame_detalles = tk.Frame(frame_resultados, bg="#FFFFFF")
        frame_detalles.pack(fill="x", padx=10)

        for i, e in enumerate(resultados['entrenudos']):
            texto = f"Entrenudo {i+1}: {e['largo_px']} px"
            if 'largo_cm' in e:
                texto += f" ({e['largo_cm']:.2f} cm)"
            tk.Label(frame_detalles, text=texto, font=("Segoe UI", 10), 
                    bg="#FFFFFF", fg="#64748B").pack(anchor="w", pady=2)

    def _generar_texto_resultados(self, resultados):
        texto = f"Alto caña (px): {resultados['alto_cana_px']}\n"
        texto += f"Cantidad Nudos: {resultados['cantidad_nudos']}\n"
        texto += f"Cantidad Entre nudos: {resultados['cantidad_entrenudos']}\n"
        if resultados['alto_cana_cm'] > 0:
            texto += f"Alto caña (cm): {resultados['alto_cana_cm']:.2f}\n"
        texto += "\nNudos:\n"
        for i, n in enumerate(resultados['nudos']):
            texto += f"  Nudo {i+1}: Largo(px): {n['largo_px']}, Ancho(px): {n['ancho_px']}"
            if 'largo_cm' in n:
                texto += f", Largo(cm): {n['largo_cm']:.2f}, Ancho(cm): {n['ancho_cm']:.2f}"
            texto += "\n"
        texto += "\nEntre nudos:\n"
        for i, e in enumerate(resultados['entrenudos']):
            texto += f"  Entre nudo {i+1}: Largo(px): {e['largo_px']}, Ancho(px): {e['ancho_px']}"
            if 'largo_cm' in e:
                texto += f", Largo(cm): {e['largo_cm']:.2f}, Ancho(cm): {e['ancho_cm']:.2f}"
            texto += "\n"
        return texto