import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import subprocess
import sys
import threading
import time

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

class MainInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Menú Principal - Proyecto MCA")
        self.root.geometry("1080x720")
        # Configuración de pantalla completa
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#0d1117")
        
        # Crear canvas para el fondo antes de todo
        self.ancho = self.root.winfo_screenwidth()
        self.alto = self.root.winfo_screenheight()
        self.canvas_bg = tk.Canvas(self.root, width=self.ancho, height=self.alto, bg="#0d1117", highlightthickness=0)
        self.canvas_bg.place(x=0, y=0, relwidth=1, relheight=1)
        # Inicializar las partículas en el fondo lo antes posible
        self.particulas = Particulas(self.canvas_bg, 120, self.ancho, self.alto)
        # Crear un frame contenedor para el contenido principal
        self.content_frame = tk.Frame(self.root, bg="#0d1117")
        self.content_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)
        self.establecer_estilos()
        self.crear_interfaz()
        
        # Set proper close handler
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.root.mainloop()

    def establecer_estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        # Colores accesibles y contrastantes
        estilo.configure("TButton", 
                        font=("Segoe UI", 13, "bold"), 
                        padding=14, 
                        foreground="#22223b", 
                        background="#2dffb3", 
                        borderwidth=0,
                        focuscolor="#2dffb3",
                        relief="flat")
        estilo.map("TButton", 
                   background=[("active", "#1ef7a3"), ("pressed", "#0d1117")],
                   foreground=[("active", "#22223b"), ("pressed", "#2dffb3")],
                   relief=[("pressed", "groove")])
        estilo.configure("TNotebook", background="#16161a", borderwidth=0)
        estilo.configure("TNotebook.Tab", font=("Segoe UI", 12, "bold"), background="#232946", foreground="#e6e6ef", padding=[18, 10])
        estilo.map("TNotebook.Tab",
                   background=[("selected", "#2dffb3"), ("active", "#1ef7a3")],
                   foreground=[("selected", "#22223b"), ("active", "#22223b")],
                   expand=[("selected", [1, 1, 1, 0])])
        estilo.configure("TFrame", background="#16161a")
        estilo.configure("TLabel", background="#16161a", foreground="#e6e6ef")
        # Animación de hover para botones
        self.root.option_add("*TButton.activeBackground", "#1ef7a3")
        self.root.option_add("*TButton.activeForeground", "#22223b")
        self.root.option_add("*TButton.relief", "flat")
        self.root.option_add("*TButton.highlightThickness", 2)
        self.root.option_add("*TButton.highlightBackground", "#2dffb3")
        self.root.option_add("*TButton.highlightColor", "#2dffb3")

    def crear_interfaz(self):
        frame = ttk.Frame(self.content_frame, padding=48, style="TFrame")
        frame.pack(expand=True)
        notebook = ttk.Notebook(frame, style="TNotebook")
        frame_dataset = ttk.Frame(notebook, style="TFrame")
        notebook.add(frame_dataset, text="Crear Dataset")
        label_dataset = ttk.Label(frame_dataset, text="Procesamiento de Imágenes", font=("Segoe UI", 20, "bold"), foreground="#2dffb3", style="TLabel")
        label_dataset.pack(pady=28)
        btn_bordes = ttk.Button(frame_dataset, text="🖼️ Bordes", command=self.abrir_bordes, style="TButton")
        btn_bordes.pack(fill="x", pady=16, padx=40)
        btn_bordes.bind("<Enter>", lambda e: btn_bordes.configure(style="Hover.TButton"))
        btn_bordes.bind("<Leave>", lambda e: btn_bordes.configure(style="TButton"))
        btn_mediciones = ttk.Button(frame_dataset, text="📏 Mediciones", command=self.abrir_mediciones, style="TButton")
        btn_mediciones.pack(fill="x", pady=16, padx=40)
        btn_mediciones.bind("<Enter>", lambda e: btn_mediciones.configure(style="Hover.TButton"))
        btn_mediciones.bind("<Leave>", lambda e: btn_mediciones.configure(style="TButton"))
        frame_ia = ttk.Frame(notebook, style="TFrame")
        notebook.add(frame_ia, text="IA")
        label_ia = ttk.Label(frame_ia, text="Modelo de Inteligencia Artificial", font=("Segoe UI", 20, "bold"), foreground="#2dffb3", style="TLabel")
        label_ia.pack(pady=28)
        btn_entrenar = ttk.Button(frame_ia, text="🚀 Entrenamiento Modelo", command=self.entrenar_modelo, style="TButton")
        btn_entrenar.pack(fill="x", pady=16, padx=40)
        btn_entrenar.bind("<Enter>", lambda e: btn_entrenar.configure(style="Hover.TButton"))
        btn_entrenar.bind("<Leave>", lambda e: btn_entrenar.configure(style="TButton"))
        btn_probar = ttk.Button(frame_ia, text="🧪 Probar modelo", command=self.probar_modelo, style="TButton")
        btn_probar.pack(fill="x", pady=16, padx=40)
        btn_probar.bind("<Enter>", lambda e: btn_probar.configure(style="Hover.TButton"))
        btn_probar.bind("<Leave>", lambda e: btn_probar.configure(style="TButton"))
        btn_reentrenar = ttk.Button(frame_ia, text="🔄 RE Entrenamiento", command=self.reentrenar_modelo, style="TButton")
        btn_reentrenar.pack(fill="x", pady=16, padx=40)
        btn_reentrenar.bind("<Enter>", lambda e: btn_reentrenar.configure(style="Hover.TButton"))
        btn_reentrenar.bind("<Leave>", lambda e: btn_reentrenar.configure(style="TButton"))
        notebook.pack(expand=True, fill="both", padx=32, pady=32)
        # Estilo hover para botones
        estilo = ttk.Style()
        estilo.configure("Hover.TButton", background="#1ef7a3", foreground="#22223b", font=("Segoe UI", 13, "bold"), padding=14, borderwidth=0)

    def abrir_bordes(self):
        try:
            # Import error could happen if the module doesn't exist
            from gui.detector_bordes_gui import DetectorBordesGUI
            self.root.destroy()
            DetectorBordesGUI()
        except ImportError:
            messagebox.showerror("Error", "No se encontró el módulo de detector de bordes.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el detector de bordes:\n{e}")

    def entrenar_modelo(self):
        try:
            # Check if model module exists
            try:
                from model.deep_sugarcane_model import train_model
            except ImportError:
                messagebox.showerror("Error", "No se encontró el módulo de entrenamiento de modelo.")
                return
                
            log_window = tk.Toplevel(self.root)
            log_window.title("Seguimiento del Entrenamiento")
            log_window.geometry("700x400")
            
            log_text = tk.Text(log_window, wrap="word", state="disabled", bg="#222", fg="#0f0", font=("Consolas", 10))
            log_text.pack(expand=True, fill="both")
            
            # Back button
            btn_back = tk.Button(log_window, text="Regresar", 
                                command=lambda: self.regresar_menu(log_window), 
                                font=("Segoe UI", 12, "bold"), bg="#64748B", fg="#FFF")
            btn_back.pack(pady=10)
            
            log_window.update()
            
            import threading, sys
            
            class ConsoleRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                def write(self, msg):
                    try:
                        self.text_widget.configure(state="normal")
                        self.text_widget.insert("end", msg)
                        self.text_widget.see("end")
                        self.text_widget.configure(state="disabled")
                    except tk.TclError:
                        # Widget might have been destroyed
                        pass
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
                    try:
                        log_text.configure(state="normal")
                        log_text.insert("end", "\nEntrenamiento completado.\n")
                        log_text.configure(state="disabled")
                        messagebox.showinfo("Entrenamiento", "Entrenamiento completado.")
                    except tk.TclError:
                        # Widget might have been destroyed
                        pass
                except Exception as e:
                    try:
                        log_text.configure(state="normal")
                        log_text.insert("end", f"\nError durante el entrenamiento:\n{e}\n")
                        log_text.configure(state="disabled")
                        messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")
                    except tk.TclError:
                        # Widget might have been destroyed
                        pass
                finally:
                    sys.stdout = original_stdout
                    
            threading.Thread(target=run_training, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")

    def obtener_modelos_disponibles(self):
        modelos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
        if not os.path.exists(modelos_dir):
            return []
        modelos = [f for f in os.listdir(modelos_dir) if f.endswith(".pth")]
        return modelos

    def reentrenar_modelo(self):
        try:
            # Check if model module exists
            try:
                from model.deep_sugarcane_model import train_model
            except ImportError:
                messagebox.showerror("Error", "No se encontró el módulo de entrenamiento de modelo.")
                return
                
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
                
                # Botón para regresar al menú principal
                btn_back = tk.Button(log_window, text="Regresar", 
                                    command=lambda: self.regresar_menu(log_window), 
                                    font=("Segoe UI", 12, "bold"), bg="#64748B", fg="#FFF")
                btn_back.pack(pady=10)
                
                log_window.update()
                
                import threading, sys
                
                class ConsoleRedirector:
                    def __init__(self, text_widget):
                        self.text_widget = text_widget
                    def write(self, msg):
                        try:
                            self.text_widget.configure(state="normal")
                            self.text_widget.insert("end", msg)
                            self.text_widget.see("end")
                            self.text_widget.configure(state="disabled")
                        except tk.TclError:
                            # Widget might have been destroyed
                            pass
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
                        try:
                            log_text.configure(state="normal")
                            log_text.insert("end", "\nRE Entrenamiento completado.\n")
                            log_text.configure(state="disabled")
                            messagebox.showinfo("RE Entrenamiento", f"Modelo seleccionado reentrenado exitosamente.")
                        except tk.TclError:
                            # Widget might have been destroyed
                            pass
                    except Exception as e:
                        try:
                            log_text.configure(state="normal")
                            log_text.insert("end", f"\nError durante el reentrenamiento:\n{e}\n")
                            log_text.configure(state="disabled")
                            messagebox.showerror("Error", f"Error durante el reentrenamiento:\n{e}")
                        except tk.TclError:
                            # Widget might have been destroyed
                            pass
                    finally:
                        sys.stdout = original_stdout
                        
                threading.Thread(target=run_retraining, daemon=True).start()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el reentrenamiento:\n{e}")

    def probar_modelo(self):
        try:
            app_path = os.path.join(os.path.dirname(__file__), "gui", "app.py")
            if os.path.exists(app_path):
                self.root.destroy()
                # Ejecutar la interfaz de clasificación
                subprocess.Popen([sys.executable, app_path])
            else:
                messagebox.showerror("Error", "No se encontró la interfaz de prueba de modelo.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la interfaz de prueba de modelo:\n{e}")

    def abrir_mediciones(self):
        try:
            from gui.mediciones_gui import MedicionesGUI
            MedicionesGUI()
        except ImportError:
            messagebox.showerror("Error", "No se encontró el módulo de mediciones.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la interfaz de Mediciones:\n{e}")

    def cerrar_aplicacion(self):
        """Cleanup and close application"""
        # Detener la animación de partículas si existe
        if hasattr(self, 'particulas'):
            self.particulas.stop()
            
        try:
            self.root.destroy()
        except:
            pass

    def regresar_menu(self, ventana=None):
        """Return to main menu and clean up properly"""
        if ventana:
            try:
                ventana.destroy()
            except:
                pass
                
        if hasattr(self, 'root') and self.root:
            try:
                self.root.destroy()
            except:
                pass
                
        # Start the main application again
        try:
            main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app_main.py'))
            subprocess.Popen([sys.executable, main_path])
        except Exception as e:
            print(f"Error al reiniciar la aplicación: {e}")

# If this file is run directly, show the presentation GUI first, then main interface
if __name__ == "__main__":
    try:
        from gui.presentacion import PresentacionGUI
        PresentacionGUI()
    except ImportError:
        # If presentation module can't be imported, start main interface directly
        MainInterface()
    except Exception as e:
        print(f"Error al iniciar la presentación: {e}")
        # Start main interface as fallback
        try:
            MainInterface()
        except Exception as e:
            print(f"Error crítico: {e}")
    def agregar_barra_ventana(self):
        barra = tk.Frame(self.root, bg="#0d1117", highlightthickness=0)
        barra.place(x=0, y=0, relwidth=1, height=48)
        # Botón Atrás
        btn_atras = tk.Button(barra, text="\U0001F519", font=("Segoe UI", 16, "bold"), bg="#232946", fg="#2dffb3", bd=0, activebackground="#1ef7a3", activeforeground="#232946", cursor="hand2", command=self.regresar_menu)
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