import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.deep_sugarcane_model import predict_image
import yaml

def get_class_names():
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset.yaml'))
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        return data['names']

class ModelTestApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Probar modelo de caña de azúcar")
        self.root.geometry("500x500")
        self.class_names = get_class_names()
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)
        self.result_label = tk.Label(self.root, text="Selecciona una imagen para probar el modelo", font=("Segoe UI", 14))
        self.result_label.pack(pady=10)
        btn_select = tk.Button(self.root, text="Seleccionar imagen", command=self.select_image, font=("Segoe UI", 12, "bold"), bg="#1E40AF", fg="#FFF")
        btn_select.pack(pady=10)
        self.root.mainloop()

    def select_image(self):
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png"),
            ("Todos los archivos", "*.*")
        ]
        img_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
        if img_path:
            try:
                img = Image.open(img_path)
                img.thumbnail((320, 320))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk
                pred = predict_image(img_path)
                class_name = self.class_names[pred] if pred < len(self.class_names) else f"Clase {pred}"
                self.result_label.config(text=f"Predicción: {class_name}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo predecir la imagen:\n{e}")

if __name__ == "__main__":
    ModelTestApp()