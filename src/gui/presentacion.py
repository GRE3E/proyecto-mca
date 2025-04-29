import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import threading
import time
import os

class Particulas:
    def __init__(self, canvas, cantidad, ancho, alto):
        self.canvas = canvas
        self.particulas = []
        self.ancho = ancho
        self.alto = alto
        self.running = True  # Flag to control animation
        
        for _ in range(cantidad):
            x = self.ancho * 0.1 + (self.ancho * 0.8) * (os.urandom(1)[0]/255)
            y = self.alto * 0.1 + (self.alto * 0.8) * (os.urandom(1)[0]/255)
            r = 2 + (os.urandom(1)[0] % 4)
            dx = (os.urandom(1)[0] - 128) / 128.0 * 0.7
            dy = (os.urandom(1)[0] - 128) / 128.0 * 0.7
            p = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#2dffb3", outline="")
            self.particulas.append({'id': p, 'x': x, 'y': y, 'r': r, 'dx': dx, 'dy': dy})
        
        self.animation_id = None  # Track animation ID
        self.animar()

    def animar(self):
        if not self.running:
            return  # Stop animation if not running
            
        for p in self.particulas:
            p['x'] += p['dx']
            p['y'] += p['dy']
            if p['x'] < 0 or p['x'] > self.ancho:
                p['dx'] *= -1
            if p['y'] < 0 or p['y'] > self.alto:
                p['dy'] *= -1
            self.canvas.coords(p['id'], p['x']-p['r'], p['y']-p['r'], p['x']+p['r'], p['y']+p['r'])
            
        # Store the ID so we can cancel it later
        self.animation_id = self.canvas.after(30, self.animar)
        
    def stop(self):
        """Stop the animation and cancel pending callbacks"""
        self.running = False
        if self.animation_id:
            try:
                self.canvas.after_cancel(self.animation_id)
                self.animation_id = None
            except:
                pass

class PresentacionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Proyecto MCA - Presentación")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#0d1117")
        self.ancho = self.root.winfo_screenwidth()
        self.alto = self.root.winfo_screenheight()
        
        self.canvas_bg = Canvas(self.root, width=self.ancho, height=self.alto, bg="#0d1117", highlightthickness=0)
        self.canvas_bg.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.frame = tk.Frame(self.root, bg="#0d1117")
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Initialize animation components
        self.particulas = Particulas(self.canvas_bg, 60, self.ancho, self.alto)
        self.halo = None
        self._color_idx = 0
        self._colores_anim = None
        self.halo_animation_id = None
        
        # Setup the UI
        self.agregar_barra_ventana()
        self.animar_holograma()
        self.mostrar_contenido()
        
        # Bind escape key and close event
        self.root.bind("<Escape>", lambda e: self.cerrar_aplicacion())
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # Start mainloop
        self.root.mainloop()

    def animar_holograma(self):
        # Efecto de halo/holograma animado
        self.halo = self.canvas_bg.create_oval(
            self.ancho//2-320, self.alto//2-220, 
            self.ancho//2+320, self.alto//2+220, 
            outline="#2dffb3", width=6
        )
        
        colores = ["#2dffb3", "#1ef7a3", "#1edfc3", "#1ecfc3", "#1ebfc3", "#1eafc3", "#1e9fc3", "#1e8fc3"]
        self._color_idx = 0
        self._colores_anim = colores + colores[::-1]
        
        self.anim_halo()
        
    def anim_halo(self):
        if not hasattr(self, 'canvas_bg') or not self.canvas_bg.winfo_exists():
            return  # Stop if canvas no longer exists
            
        color = self._colores_anim[self._color_idx % len(self._colores_anim)]
        try:
            self.canvas_bg.itemconfig(self.halo, outline=color)
            self._color_idx = (self._color_idx + 1) % len(self._colores_anim)
            # Store the ID so we can cancel it later
            self.halo_animation_id = self.canvas_bg.after(40, self.anim_halo)
        except tk.TclError:
            # Canvas might be destroyed already
            pass

    def mostrar_contenido(self):
        # Cargar el README.md
        readme_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "README.md"))
        contenido = ""
        if os.path.exists(readme_path):
            try:
                with open(readme_path, encoding="utf-8") as f:
                    contenido = f.read()
            except:
                # If README can't be loaded, use default text
                contenido = "## Descripción\nSistema inteligente para clasificación de imágenes de caña de azúcar.\n## Requisitos"
        
        # Título principal
        titulo = tk.Label(self.frame, text="Sistema de Clasificación de Caña de Azúcar", 
                          font=("Segoe UI", 38, "bold"), fg="#2dffb3", bg="#0d1117")
        titulo.pack(pady=(0, 20))
        
        subtitulo = tk.Label(self.frame, text="Proyecto MCA", 
                            font=("Segoe UI", 22, "bold"), fg="#e6e6ef", bg="#0d1117")
        subtitulo.pack(pady=(0, 10))
        
        # Características principales
        caracteristicas = [
            "\u25CF Procesamiento avanzado de imágenes (OpenCV, NumPy)",
            "\u25CF Detección de bordes y extracción de regiones de interés (Scikit-Image)",
            "\u25CF Clasificación mediante aprendizaje profundo (PyTorch, TorchVision)",
            "\u25CF Interfaz intuitiva y visualización en tiempo real (Matplotlib)",
            "\u25CF Entrenamiento y reentrenamiento de modelos",
            "\u25CF Resultados precisos y visualmente atractivos"
        ]
        
        for c in caracteristicas:
            tk.Label(self.frame, text=c, font=("Segoe UI", 16), fg="#a1a1b5", bg="#0d1117").pack(anchor="w", padx=30)
        
        # Separador
        tk.Frame(self.frame, height=2, bg="#30363d").pack(fill="x", pady=20)
        
        # Descripción del README
        desc = self.extraer_descripcion(contenido)
        desc_label = tk.Label(self.frame, text=desc, font=("Segoe UI", 14), fg="#e6e6ef", 
                             bg="#0d1117", wraplength=self.ancho*0.7, justify="left")
        desc_label.pack(pady=(0, 20))
        
        # Botón para continuar
        btn = tk.Button(self.frame, text="Entrar al sistema", font=("Segoe UI", 18, "bold"), 
                       fg="#0d1117", bg="#2dffb3", activebackground="#1ef7a3", activeforeground="#0d1117", 
                       bd=0, padx=30, pady=10, command=self.abrir_menu)
        btn.pack(pady=30)
        
        btn.bind("<Enter>", lambda e: btn.config(bg="#1ef7a3"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#2dffb3"))
        
        # Animación de entrada
        self.frame.after(100, lambda: self.frame.tkraise())

    def extraer_descripcion(self, contenido):
        # Extrae la sección de descripción del README
        inicio = contenido.find("## Descripción")
        fin = contenido.find("## Requisitos")
        
        if inicio != -1 and fin != -1:
            return contenido[inicio+15:fin].strip()
        
        return "Sistema inteligente para clasificación de imágenes de caña de azúcar."

    def abrir_menu(self):
        # Clean up resources before destroying the window
        self.cerrar_aplicacion(abrir_menu=True)
        
    def cerrar_aplicacion(self, abrir_menu=False):
        """Cleanup resources and close application"""
        # Stop animations
        if hasattr(self, 'particulas'):
            self.particulas.stop()
            
        # Cancel halo animation
        if self.halo_animation_id:
            try:
                self.canvas_bg.after_cancel(self.halo_animation_id)
                self.halo_animation_id = None
            except:
                pass
        
        # Destroy the root window
        try:
            self.root.destroy()
        except:
            pass
            
        # Optionally open the main interface
        if abrir_menu:
            try:
                from app_main import MainInterface
                MainInterface()
            except Exception as e:
                print(f"Error al abrir el menú principal: {e}")

    def agregar_barra_ventana(self):
        barra = tk.Frame(self.root, bg="#0d1117", highlightthickness=0)
        barra.place(x=0, y=0, relwidth=1, height=48)
        # Botón Atrás
        btn_atras = tk.Button(barra, text="\U0001F519", font=("Segoe UI", 16, "bold"), bg="#232946", fg="#2dffb3", bd=0, activebackground="#1ef7a3", activeforeground="#232946", cursor="hand2", command=self.abrir_menu)
        btn_atras.pack(side="left", padx=(16,8), pady=8)
        btn_atras.bind("<Enter>", lambda e: btn_atras.config(bg="#1ef7a3", fg="#232946"))
        btn_atras.bind("<Leave>", lambda e: btn_atras.config(bg="#232946", fg="#2dffb3"))
        # Botón Minimizar
        btn_min = tk.Button(barra, text="\u2796", font=("Segoe UI", 16, "bold"), bg="#232946", fg="#2dffb3", bd=0, activebackground="#1ef7a3", activeforeground="#232946", cursor="hand2", command=self.minimizar_ventana)
        btn_min.pack(side="right", padx=(8,8), pady=8)
        btn_min.bind("<Enter>", lambda e: btn_min.config(bg="#1ef7a3", fg="#232946"))
        btn_min.bind("<Leave>", lambda e: btn_min.config(bg="#232946", fg="#2dffb3"))
        # Botón Cerrar
        btn_cerrar = tk.Button(barra, text="\u274C", font=("Segoe UI", 16, "bold"), bg="#232946", fg="#2dffb3", bd=0, activebackground="#ff4b4b", activeforeground="#fff", cursor="hand2", command=self.cerrar_aplicacion)
        btn_cerrar.pack(side="right", padx=(8,16), pady=8)
        btn_cerrar.bind("<Enter>", lambda e: btn_cerrar.config(bg="#ff4b4b", fg="#fff"))
        btn_cerrar.bind("<Leave>", lambda e: btn_cerrar.config(bg="#232946", fg="#2dffb3"))

    def minimizar_ventana(self):
        self.root.iconify()
        
if __name__ == "__main__":
    PresentacionGUI()