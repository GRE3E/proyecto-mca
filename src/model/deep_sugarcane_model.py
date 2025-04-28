import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

class SugarCaneDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_measurements=15):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.max_measurements = max_measurements  # Máximo número de mediciones a considerar
        
        # Leer clases desde dataset.yaml
        import yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dataset.yaml')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data['names']
            
        # Solo considerar carpetas que estén en class_names
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                        img_path = os.path.join(class_path, fname)
                        json_path = os.path.splitext(img_path)[0] + '.json'
                        
                        if os.path.exists(json_path):
                            with open(json_path, 'r') as json_file:
                                data = json.load(json_file)
                                
                                # Nuevo formato JSON
                                if "Caña" in data and "Mediciones" in data:
                                    cane_data = data["Caña"]
                                    measurements = data["Mediciones"]
                                    
                                    # Extraer altura
                                    height_straight = cane_data.get("AltoRecto_cm", 0)
                                    height_curved = cane_data.get("AltoCurvo_cm", 0)
                                    
                                    # Procesar mediciones (hasta max_measurements)
                                    processed_measurements = []
                                    for i, m in enumerate(measurements):
                                        if i >= self.max_measurements:
                                            break
                                        measurement = [
                                            float(m.get("Nudo_Largo_cm", 0)),
                                            float(m.get("Nudo_Ancho_cm", 0)),
                                            float(m.get("Entrenudo_Largo_cm", 0)),
                                            float(m.get("Entrenudo_Ancho_cm", 0))
                                        ]
                                        processed_measurements.append(measurement)
                                    
                                    # Rellenar con ceros hasta tener max_measurements
                                    while len(processed_measurements) < self.max_measurements:
                                        processed_measurements.append([0.0, 0.0, 0.0, 0.0])
                                    
                                    # Aplanar para facilitar el procesamiento
                                    flattened_measurements = [item for sublist in processed_measurements for item in sublist]
                                    
                                    self.samples.append((
                                        img_path,
                                        label,
                                        height_straight,
                                        height_curved,
                                        flattened_measurements
                                    ))
                                # Formato antiguo (compatibilidad)
                                else:
                                    measures = data.get('Mediciones', [0, 0, 0, 0])
                                    if isinstance(measures, list):
                                        if all(isinstance(m, dict) for m in measures):
                                            first_measure = measures[0] if measures else {}
                                            flattened = [
                                                float(first_measure.get("Nudo_Largo_cm", 0)),
                                                float(first_measure.get("Nudo_Ancho_cm", 0)),
                                                float(first_measure.get("Entrenudo_Largo_cm", 0)),
                                                float(first_measure.get("Entrenudo_Ancho_cm", 0))
                                            ]
                                        else:
                                            flattened = [float(m) for m in measures[:4]]
                                            flattened.extend([0.0] * (4 - len(flattened)))
                                    else:
                                        flattened = [float(measures), 0.0, 0.0, 0.0]
                                    
                                    # Rellenar para mantener consistencia con el nuevo formato
                                    dummy_measurements = [flattened]
                                    while len(dummy_measurements) < self.max_measurements:
                                        dummy_measurements.append([0.0, 0.0, 0.0, 0.0])
                                    
                                    flattened_dummy = [item for sublist in dummy_measurements for item in sublist]
                                    
                                    self.samples.append((
                                        img_path,
                                        label,
                                        0.0,  # No hay info de altura
                                        0.0,  # No hay info de altura
                                        flattened_dummy
                                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, height_straight, height_curved, flattened_measurements = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convertir a tensores
        height_tensor = torch.tensor([height_straight, height_curved], dtype=torch.float32)
        measures_tensor = torch.tensor(flattened_measurements, dtype=torch.float32)
        
        return image, label, height_tensor, measures_tensor

class DeepSugarCaneNet(nn.Module):
    def __init__(self, num_classes=2, max_measurements=15):
        super().__init__()
        self.max_measurements = max_measurements
        
        # Extractor de características - usar ResNet pre-entrenado
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-2]  # Quitar las últimas capas
        self.features = nn.Sequential(*modules)
        
        # Capas de atención para mejorar la extracción de características
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Características combinadas tras la atención
        self.attention_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Clasificador para determinar si es caña o no
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Predicción de altura (AltoRecto_cm, AltoCurvo_cm)
        self.height_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Regresión para medidas de nudos y entrenudos (múltiples secciones)
        # 4 medidas por sección × max_measurements secciones
        self.measurements = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * max_measurements)
        )
        
    def forward(self, x):
        # Extracción de características
        features = self.features(x)
        
        # Mecanismo de atención
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        
        # Características globales con atención
        global_features = self.attention_features(weighted_features)
        
        # Salidas
        class_output = self.classifier(global_features)
        height_output = self.height_regressor(global_features)
        measurements_output = self.measurements(global_features)
        
        return class_output, height_output, measurements_output

def normalize_pixel_to_cm(image, pixel_to_cm_ratio=10.0):
    """
    Normaliza las dimensiones de la imagen considerando la relación píxel-cm.
    
    Args:
        image: Imagen a normalizar (array numpy o PIL Image)
        pixel_to_cm_ratio: Relación píxeles/cm (por defecto: 10.0)
    
    Returns:
        La imagen normalizada y el factor de escala aplicado
    """
    import numpy as np
    from PIL import Image
    
    # Convertir a numpy array si es una imagen PIL
    if isinstance(image, Image.Image):
        np_image = np.array(image)
    else:
        np_image = image.copy()
    
    # Dimensiones originales de la imagen
    if len(np_image.shape) == 3:
        height, width, _ = np_image.shape
    else:
        height, width = np_image.shape
    
    # Calcular ancho y alto en cm basado en la relación pixel_to_cm_ratio
    width_cm = width / pixel_to_cm_ratio
    height_cm = height / pixel_to_cm_ratio
    
    # Si necesitáramos escalar la imagen a un tamaño estándar en cm,
    # aquí calcularíamos el factor de escala y redimensionaríamos
    # Por ahora, solo pasamos los metadatos junto con la imagen original
    
    return {
        'image': np_image,
        'width_px': width,
        'height_px': height,
        'width_cm': width_cm,
        'height_cm': height_cm,
        'px_to_cm_ratio': pixel_to_cm_ratio
    }
def get_dataloaders(train_dir, val_dir, batch_size=16, img_size=320, max_measurements=15):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = SugarCaneDataset(train_dir, transform, max_measurements)
    val_ds = SugarCaneDataset(val_dir, val_transform, max_measurements)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train_model(train_dir, val_dir, model_path=None, epochs=30, batch_size=16, img_size=320, 
                max_measurements=15, lr=1e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size, img_size, max_measurements)
    print(f"Conjuntos de datos cargados: {len(train_loader.dataset)} imágenes de entrenamiento, "
          f"{len(val_loader.dataset)} imágenes de validación")
    
    if model_path and os.path.exists(model_path):
        print(f"Cargando modelo desde {model_path}")
        model = DeepSugarCaneNet(num_classes=2, max_measurements=max_measurements).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Creando nuevo modelo")
        model = DeepSugarCaneNet(num_classes=2, max_measurements=max_measurements).to(device)
    
    # Criterios de pérdida para clasificación y regresión
    class_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    # Optimizador con learning rate diferentes para características y cabezas
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': lr/10},
        {'params': model.classifier.parameters()},
        {'params': model.height_regressor.parameters()},
        {'params': model.measurements.parameters()}
    ], lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = 7  # Early stopping patience
    
    history = {
        'train_class_loss': [], 'train_height_loss': [], 'train_measure_loss': [], 'train_total_loss': [],
        'val_class_loss': [], 'val_height_loss': [], 'val_measure_loss': [], 'val_total_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        running_class_loss = 0.0
        running_height_loss = 0.0
        running_measure_loss = 0.0
        running_total_loss = 0.0
        total_batches = len(train_loader)
        
        print(f"\nÉpoca {epoch+1}/{epochs}")
        print("-" * 50)
        
        for batch_idx, (images, labels, heights, measures) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            heights = heights.to(device)
            measures = measures.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_outputs, height_outputs, measure_outputs = model(images)
            
            # Calcular pérdidas
            class_loss = class_criterion(class_outputs, labels)
            
            # Solo calcular pérdidas de regresión para imágenes de caña real (label=1)
            if torch.sum(labels == 1) > 0:
                # Máscara para seleccionar solo las cañas
                cane_mask = (labels == 1)
                height_loss = regression_criterion(
                    height_outputs[cane_mask], 
                    heights[cane_mask]
                )
                measure_loss = regression_criterion(
                    measure_outputs[cane_mask], 
                    measures[cane_mask]
                )
            else:
                height_loss = torch.tensor(0.0, device=device)
                measure_loss = torch.tensor(0.0, device=device)
            
            # Pérdida total (ponderada para balancear clasificación y regresión)
            total_loss = class_loss + 0.5 * height_loss + 0.5 * measure_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Actualizar estadísticas
            running_class_loss += class_loss.item() * images.size(0)
            running_height_loss += height_loss.item() * images.size(0)
            running_measure_loss += measure_loss.item() * images.size(0)
            running_total_loss += total_loss.item() * images.size(0)
            
            # Imprimir progreso cada 5 batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                print(f"Batch {batch_idx+1}/{total_batches} - "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Class: {class_loss.item():.4f}, "
                      f"Height: {height_loss.item():.4f}, "
                      f"Measures: {measure_loss.item():.4f})")
        
        # Calcular pérdidas promedio
        train_class_loss = running_class_loss / len(train_loader.dataset)
        train_height_loss = running_height_loss / len(train_loader.dataset)
        train_measure_loss = running_measure_loss / len(train_loader.dataset)
        train_total_loss = running_total_loss / len(train_loader.dataset)
        
        # Evaluación
        val_metrics = evaluate(model, val_loader, class_criterion, regression_criterion, device)
        val_class_loss, val_height_loss, val_measure_loss, val_total_loss, val_acc = val_metrics
        
        # Learning rate scheduler
        scheduler.step(val_total_loss)
        
        # Actualizar historial
        history['train_class_loss'].append(train_class_loss)
        history['train_height_loss'].append(train_height_loss)
        history['train_measure_loss'].append(train_measure_loss)
        history['train_total_loss'].append(train_total_loss)
        history['val_class_loss'].append(val_class_loss)
        history['val_height_loss'].append(val_height_loss)
        history['val_measure_loss'].append(val_measure_loss)
        history['val_total_loss'].append(val_total_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nResumen Época {epoch+1}/{epochs}:")
        print(f"Train - Total Loss: {train_total_loss:.4f} (Class: {train_class_loss:.4f}, "
              f"Height: {train_height_loss:.4f}, Measures: {train_measure_loss:.4f})")
        print(f"Val - Total Loss: {val_total_loss:.4f} (Class: {val_class_loss:.4f}, "
              f"Height: {val_height_loss:.4f}, Measures: {val_measure_loss:.4f}), Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            no_improve_epochs = 0
            best_model_path = os.path.join(os.path.dirname(__file__), 'best_sugarcane_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Mejor modelo guardado en {best_model_path}")
        else:
            no_improve_epochs += 1
            
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping después de {patience} épocas sin mejora")
            break
    
    plot_metrics(history)
    return model, history

def evaluate(model, loader, class_criterion, regression_criterion, device):
    model.eval()
    class_loss = 0.0
    height_loss = 0.0
    measure_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, heights, measures in loader:
            images = images.to(device)
            labels = labels.to(device)
            heights = heights.to(device)
            measures = measures.to(device)
            
            # Forward pass
            class_outputs, height_outputs, measure_outputs = model(images)
            
            # Calcular pérdidas
            class_loss_batch = class_criterion(class_outputs, labels)
            
            # Solo calcular pérdidas de regresión para imágenes de caña real (label=1)
            if torch.sum(labels == 1) > 0:
                cane_mask = (labels == 1)
                height_loss_batch = regression_criterion(
                    height_outputs[cane_mask], 
                    heights[cane_mask]
                )
                measure_loss_batch = regression_criterion(
                    measure_outputs[cane_mask], 
                    measures[cane_mask]
                )
            else:
                height_loss_batch = torch.tensor(0.0, device=device)
                measure_loss_batch = torch.tensor(0.0, device=device)
            
            # Pérdida total (ponderada igual que en entrenamiento)
            total_loss_batch = class_loss_batch + 0.5 * height_loss_batch + 0.5 * measure_loss_batch
            
            # Acumular pérdidas
            class_loss += class_loss_batch.item() * images.size(0)
            height_loss += height_loss_batch.item() * images.size(0)
            measure_loss += measure_loss_batch.item() * images.size(0)
            total_loss += total_loss_batch.item() * images.size(0)
            
            # Calcular precisión de clasificación
            _, preds = torch.max(class_outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Calcular promedios
    dataset_size = len(loader.dataset)
    avg_class_loss = class_loss / dataset_size
    avg_height_loss = height_loss / dataset_size
    avg_measure_loss = measure_loss / dataset_size
    avg_total_loss = total_loss / dataset_size
    acc = correct / total
    
    return avg_class_loss, avg_height_loss, avg_measure_loss, avg_total_loss, acc

def plot_metrics(history, save_dir=None):
    """Grafica las métricas de entrenamiento y validación."""
    plt.figure(figsize=(15, 10))
    
    # Pérdidas de entrenamiento
    plt.subplot(2, 2, 1)
    plt.plot(history['train_class_loss'], label='Class Loss')
    plt.plot(history['train_height_loss'], label='Height Loss')
    plt.plot(history['train_measure_loss'], label='Measure Loss')
    plt.plot(history['train_total_loss'], label='Total Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Pérdidas de validación
    plt.subplot(2, 2, 2)
    plt.plot(history['val_class_loss'], label='Class Loss')
    plt.plot(history['val_height_loss'], label='Height Loss')
    plt.plot(history['val_measure_loss'], label='Measure Loss')
    plt.plot(history['val_total_loss'], label='Total Loss')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Precisión de validación
    plt.subplot(2, 2, 3)
    plt.plot(history['val_acc'], label='Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Comparación de pérdidas totales
    plt.subplot(2, 2, 4)
    plt.plot(history['train_total_loss'], label='Training')
    plt.plot(history['val_total_loss'], label='Validation')
    plt.title('Total Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(save_path)
        print(f"Gráficas guardadas en {save_path}")
    else:
        plt.show()
    
    plt.close()

def predict_image(image_path, model_path=None, device=None, max_measurements=15, pixel_to_cm_ratio=10.0):
    """
    Predice si una imagen es de caña de azúcar y sus características, aplicando la relación píxel-cm.
    
    Args:
        image_path: Ruta de la imagen a analizar
        model_path: Ruta al modelo preentrenado (opcional)
        device: Dispositivo de computación ('cuda' o 'cpu')
        max_measurements: Número máximo de mediciones a considerar
        pixel_to_cm_ratio: Relación píxeles/cm (por defecto: 10.0)
        
    Returns:
        Diccionario con los resultados de la predicción y mediciones
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo
    model = DeepSugarCaneNet(num_classes=2, max_measurements=max_measurements)
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'best_sugarcane_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    # Preprocesamiento de imagen
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Normalizar imagen con la relación píxel-cm
    image_info = normalize_pixel_to_cm(Image.open(image_path).convert('RGB'), pixel_to_cm_ratio)
    original_ratio = pixel_to_cm_ratio / 10.0  # Factor de ajuste respecto al entrenamiento
    
    # Transformar para la red
    image_tensor = transform(Image.fromarray(image_info['image']) if isinstance(image_info['image'], np.ndarray) 
                           else Image.open(image_path).convert('RGB'))
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        class_outputs, height_outputs, measure_outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(class_outputs, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)
        
        # Convertir a formatos Python nativos
        es_cana = bool(pred_class.item() == 1)
        confidence_value = float(confidence.item())
        
        # Si es una caña de azúcar (clase 1), procesar las medidas
        if es_cana:
            # Obtener medidas de altura y ajustar según la relación píxel-cm
            alto_recto, alto_curvo = height_outputs[0].cpu().numpy()
            
            # Aplicar el factor de escala para ajustar las mediciones al ratio actual
            alto_recto = float(alto_recto / original_ratio)
            alto_curvo = float(alto_curvo / original_ratio)
            
            # Obtener mediciones de nudos y entrenudos
            measures_flat = measure_outputs[0].cpu().numpy()
            
            # Reorganizar en grupos de 4 valores
            measures_grouped = []
            for i in range(0, len(measures_flat), 4):
                if i + 4 <= len(measures_flat):
                    group = measures_flat[i:i+4]
                    # Aplicar el factor de escala a cada medida
                    adjusted_group = [float(val / original_ratio) for val in group]
                    
                    # Sólo incluir mediciones no nulas
                    if not all(v == 0 for v in adjusted_group):
                        measures_grouped.append({
                            "Nudo_Largo_cm": adjusted_group[0],
                            "Nudo_Ancho_cm": adjusted_group[1],
                            "Entrenudo_Largo_cm": adjusted_group[2],
                            "Entrenudo_Ancho_cm": adjusted_group[3]
                        })
            
            # Crear resultado en formato JSON esperado
            return {
                "Caña": {
                    "AltoRecto_cm": alto_recto,
                    "AltoCurvo_cm": alto_curvo,
                    "Confianza": confidence_value
                },
                "Mediciones": measures_grouped
            }
        else:
            # No es caña
            return {
                "Caña": False,
                "Confianza": confidence_value
            }
if __name__ == "__main__":
    # Ejemplo de uso
    print("DeepSugarCaneNet - Modelo para detección y medición de caña de azúcar")
    print("Utilizar train_model() para entrenar o predict_image() para predecir")