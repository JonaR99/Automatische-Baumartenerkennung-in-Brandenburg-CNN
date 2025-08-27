# Testregion Evaluation - Baumarten-Klassifikation und Validierung
# Wendet das beste Modell auf 17.500 Tiles an und vergleicht mit Shapefile

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
# ============================================
# KONFIGURATION
# ============================================
# Pfade anpassen!
MODEL_PATH = 'D:\\BA Jona\\training_results_20250818_140051\\best_model.pt'  # Pfad zum gespeicherten Modell
TILES_DIR = 'D:\\BA Jona\\region_1\\tiles'  # Ordner mit den 17.500 Tiles
SHAPEFILE_PATH = 'D:\\BA Jona\\region_1\\reference_data_region_1.gpkg'  # Referenz Shapefile
OUTPUT_DIR = 'D:\\BA Jona\\region_1\\results'  # Ausgabeordner

# Erstelle Ausgabeordner
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Klassen-Mapping (wie im Training)
classes = ['birke', 'buche', 'eiche', 'kiefer', 'fichte']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

DROPOUT_RATE = 0.2

print(f"Klassen-Mapping: {class_to_idx}")

# ============================================
# MODELL DEFINITION (SimplerCNN wie im Training)
# ============================================
# class SimplerCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(SimplerCNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d(0.2)
        
#         self.fc1 = nn.Linear(64 * 12 * 12, 256)
#         self.fc2 = nn.Linear(256, num_classes)
#         self.dropout_fc = nn.Dropout(0.2)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout(x)
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)

# ============================================
# MODELL DEFINITION (ImprovedCNN wie im Training)
# ============================================

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedCNN, self).__init__()
        
        # Feature Extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
       
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(DROPOUT_RATE)
        
        # Adaptive pooling für flexible Eingabegrößen
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
       
        # Classifier
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten und Fully Connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# ============================================
# DATASET FÜR TEST TILES
# ============================================
class TestTilesDataset(Dataset):
    def __init__(self, tiles_dir, transform=None):
        self.tiles_dir = tiles_dir
        self.transform = transform
        
        # Sammle alle Tile-Dateien
        self.tile_files = []
        for file in os.listdir(tiles_dir):
            if file.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                self.tile_files.append(file)
        
        print(f"Gefunden: {len(self.tile_files)} Tiles")
        
    def __len__(self):
        return len(self.tile_files)
    
    def __getitem__(self, idx):
        tile_path = os.path.join(self.tiles_dir, self.tile_files[idx])
        
        # Lade Tile
        with rasterio.open(tile_path) as src:
            image = src.read()
            # Speichere Georeferenzierung für später
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
        # Verarbeite nur RGB
        image = image[:3, :, :]
        image = ToTensor()(image)
        image = torch.permute(image, (1, 2, 0))
        
        # Resize auf 100x100 (falls nicht schon)
        resize_transform = Resize((100, 100))
        image = resize_transform(image)

        image = image.float()
        
        # Normalisierung (wie im Training)
        if self.transform is not None:
            image = self.transform(image)
        
        # Gebe auch Metadaten zurück
        return image, self.tile_files[idx], np.array([float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)])

# ============================================
# MODELL LADEN
# ============================================
print("\nLade Modell...")
model = ImprovedCNN(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Modell erfolgreich geladen!")

# ============================================
# VORHERSAGE AUF TILES
# ============================================
print("\nStarte Vorhersage auf Testregion...")

# Transform für Test (nur Normalisierung, keine Augmentation)
transform_test = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset und DataLoader
test_dataset = TestTilesDataset(TILES_DIR, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Vorhersage
predictions = []
tile_names = []
##tile_bounds = []
confidences = []

##minx_list, miny_list, maxx_list, maxy_list = [], [], [], []

geoms = []
with torch.no_grad():
    for batch_images, batch_names, batch_bounds in tqdm(test_loader, desc="Klassifiziere Tiles"):
        batch_images = batch_images.to(device)
        
        # Vorhersage
        outputs = model(batch_images)
        probs = torch.exp(outputs)  # Von log_softmax zu Wahrscheinlichkeiten
        
        # Klasse mit höchster Wahrscheinlichkeit
        max_probs, predicted = torch.max(probs, 1)
        
        # Speichere Ergebnisse
        predictions.extend(predicted.cpu().numpy())
        confidences.extend(max_probs.cpu().numpy())
        tile_names.extend(batch_names)

        # Bounds sind komplexer zu handhaben
        for bound in batch_bounds:
            if hasattr(bound, 'tolist'):
                bound = bound.tolist()
            if isinstance(bound[0], (list, tuple, np.ndarray)):
                bound = bound[0]
            if len(bound) != 4:
                print(f"FEHLER: bound hat {len(bound)} Werte: {bound}")
                continue
            geoms.append(box(*bound))
            ##minx_list.append(bound[0])
            ##miny_list.append(bound[1])
            ##maxx_list.append(bound[2])
            ##maxy_list.append(bound[3])

print(f"\nVorhersage abgeschlossen für {len(predictions)} Tiles")

# ============================================
# ERSTELLE ERGEBNIS-DATAFRAME
# ============================================

##print(f"tile_names: {len(tile_names)}")
##print(f"predictions: {len(predictions)}")
##print(f"confidences: {len(confidences)}")
##print(f"tile_bounds: {len(tile_bounds)}")

results_df = pd.DataFrame({
    'tile_name': tile_names,
    'predicted_class_idx': predictions,
    'predicted_class': [idx_to_class[idx] for idx in predictions],
    'confidence': confidences,
    'geometry': geoms
})

# Erstelle Geometrie für räumlichen Abgleich
##from shapely.geometry import box
##results_df['geometry'] = [
##    box(*bound) for bound in zip(minx_list, miny_list, maxx_list, maxy_list)
##]

# Konvertiere zu GeoDataFrame
results_gdf = gpd.GeoDataFrame(results_df, geometry='geometry')

# Setze CRS (anpassen je nach deinem Koordinatensystem!)
# Typisch für Brandenburg: EPSG:25833 (ETRS89 / UTM zone 33N)
results_gdf.crs = "EPSG:25833"  # ANPASSEN falls anders!

print("\nErgebnis-Statistik:")
print(results_df['predicted_class'].value_counts())
print(f"\nDurchschnittliche Confidence: {results_df['confidence'].mean():.3f}")

# ============================================
# LADE REFERENZ-SHAPEFILE
# ============================================
print("\nLade Referenz-Shapefile...")
reference_gdf = gpd.read_file(SHAPEFILE_PATH)

# Check CRS
if reference_gdf.crs != results_gdf.crs:
    print(f"CRS-Anpassung: {reference_gdf.crs} -> {results_gdf.crs}")
    reference_gdf = reference_gdf.to_crs(results_gdf.crs)

print(f"Referenz-Shapefile geladen: {len(reference_gdf)} Features")
print(f"Verfügbare Spalten: {reference_gdf.columns.tolist()}")

# ============================================
# RÄUMLICHER JOIN
# ============================================
print("\nFühre räumlichen Join durch...")

# WICHTIG: Passe den Spaltennamen an!
# Typische Namen könnten sein: 'baumart', 'tree_species', 'species', 'BA', etc.
BAUMART_COLUMN = 'DomBA'  # ANPASSEN an deine Shapefile-Spalte!

if BAUMART_COLUMN not in reference_gdf.columns:
    print(f"WARNUNG: Spalte '{BAUMART_COLUMN}' nicht gefunden!")
    print(f"Verfügbare Spalten: {reference_gdf.columns.tolist()}")
    print("Bitte passe BAUMART_COLUMN an!")
    # Versuche automatisch zu finden
    possible_columns = [col for col in reference_gdf.columns if 'baum' in col.lower() or 'tree' in col.lower() or 'species' in col.lower()]
    if possible_columns:
        BAUMART_COLUMN = possible_columns[0]
        print(f"Verwende Spalte: {BAUMART_COLUMN}")

# Spatial Join - findet für jedes Tile die überlappende Referenz-Baumart
joined_gdf = gpd.sjoin(results_gdf, reference_gdf[[BAUMART_COLUMN, 'geometry']], 
                        how='left', predicate='intersects')

# Manche Tiles könnten mehrere Referenz-Polygone überlappen
# Nehme das erste (oder häufigste) wenn mehrere
joined_gdf = joined_gdf.groupby(joined_gdf.index).first()

# Standardisiere Baumart-Namen (falls nötig)
# z.B. wenn im Shapefile "Gemeine Kiefer" statt "kiefer" steht
baumart_mapping = {
    'Gemeine Kiefer': 'kiefer',
    'Gemeine Fichte': 'fichte',
    ##'Pinus sylvestris': 'kiefer',
    ##'GKI': 'kiefer',
    'Rotbuche': 'buche',
    ##'Fagus sylvatica': 'buche',
    ##'RBU': 'buche',
    'Stieleiche': 'eiche',
    'Traubeneiche': 'eiche',
    'Roteiche': 'eiche',
    ##'Quercus robur': 'eiche',
    ##'SEI': 'eiche',
    'Birke': 'birke',
    ##'Betula pendula': 'birke',
    ##'GBI': 'birke',
    ##'Rot-Erle': 'rer',
    ##'Roterle': 'rer',
    ##'Alnus glutinosa': 'rer',
    ##'RER': 'rer'
}

# Wende Mapping an
joined_gdf['reference_class'] = joined_gdf[BAUMART_COLUMN].map(
    lambda x: baumart_mapping.get(x, str(x).lower()) if pd.notna(x) else None
)

# ============================================
# VALIDIERUNG
# ============================================
print("\nValidierung der Vorhersagen...")

# Filtere nur Tiles mit Referenzdaten
validated_df = joined_gdf[joined_gdf['reference_class'].notna()].copy()

# Bounds-Spalten aus der Geometrie extrahieren (NUR für die validierten Tiles)
bounds = validated_df.geometry.bounds
validated_df['minx'] = bounds['minx']
validated_df['miny'] = bounds['miny']
validated_df['maxx'] = bounds['maxx']
validated_df['maxy'] = bounds['maxy']

# Bounds-Spalten aus der Geometrie extrahieren
##validated_df['minx'] = validated_df.geometry.bounds['minx']
##validated_df['miny'] = validated_df.geometry.bounds['miny']
##validated_df['maxx'] = validated_df.geometry.bounds['maxx']
##validated_df['maxy'] = validated_df.geometry.bounds['maxy']

print(f"Tiles mit Referenzdaten: {len(validated_df)} von {len(joined_gdf)}")

if len(validated_df) > 0:
    # Berechne Accuracy
    validated_df['correct'] = validated_df['predicted_class'] == validated_df['reference_class']
    overall_accuracy = validated_df['correct'].mean()
    
    print(f"\n{'='*50}")
    print(f"GESAMTGENAUIGKEIT: {overall_accuracy:.2%}")
    print(f"{'='*50}")
    
    # Klassenweise Accuracy
    print("\nKlassenspezifische Genauigkeit:")
    for class_name in classes:
        class_mask = validated_df['reference_class'] == class_name
        if class_mask.sum() > 0:
            class_acc = validated_df[class_mask]['correct'].mean()
            class_count = class_mask.sum()
            print(f"  {class_name:10s}: {class_acc:6.2%} ({class_count} Tiles)")
    
    # ============================================
    # CONFUSION MATRIX
    # ============================================
    y_true = validated_df['reference_class']
    y_pred = validated_df['predicted_class']
    
    # Stelle sicher, dass alle Klassen vorhanden sind
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Testregion\nGesamtgenauigkeit: {overall_accuracy:.2%}')
    plt.ylabel('Referenz (Shapefile)')
    plt.xlabel('Vorhersage (CNN)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_testregion.png'), dpi=150)
    plt.show()
    
    # Classification Report
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Speichere Report
    with open(os.path.join(OUTPUT_DIR, 'classification_report_testregion.txt'), 'w') as f:
        f.write(f"Testregion Evaluation\n")
        f.write(f"="*50 + "\n")
        f.write(f"Gesamtgenauigkeit: {overall_accuracy:.2%}\n")
        f.write(f"Anzahl validierte Tiles: {len(validated_df)}\n")
        f.write(f"Anzahl Tiles ohne Referenz: {len(joined_gdf) - len(validated_df)}\n\n")
        f.write(report)
    
    # ============================================
    # RÄUMLICHE VISUALISIERUNG
    # ============================================
    print("\nErstelle räumliche Visualisierung...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Vorhergesagte Klassen
    validated_df.plot(column='predicted_class', 
                      categorical=True,
                      legend=True,
                      ax=axes[0],
                      cmap='Set3')
    axes[0].set_title('CNN Vorhersagen')
    axes[0].set_xlabel('X-Koordinate')
    axes[0].set_ylabel('Y-Koordinate')
    
    # Plot 2: Korrekt/Falsch
    colors = ['red', 'green']
    validated_df.plot(column='correct',
                      categorical=True,
                      legend=True,
                      ax=axes[1],
                      cmap=matplotlib.colors.ListedColormap(colors))
    axes[1].set_title(f'Validierung (Grün=Korrekt, Rot=Falsch)\nGenauigkeit: {overall_accuracy:.2%}')
    axes[1].set_xlabel('X-Koordinate')
    axes[1].set_ylabel('Y-Koordinate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_validation.png'), dpi=150)
    plt.show()
    
    # ============================================
    # CONFIDENCE ANALYSE
    # ============================================
    print("\nConfidence-Analyse:")
    print(f"Durchschnittliche Confidence (korrekt): {validated_df[validated_df['correct']]['confidence'].mean():.3f}")
    print(f"Durchschnittliche Confidence (falsch): {validated_df[~validated_df['correct']]['confidence'].mean():.3f}")
    
    # Plot Confidence-Verteilung
    plt.figure(figsize=(10, 6))
    plt.hist(validated_df[validated_df['correct']]['confidence'], 
             bins=30, alpha=0.5, label='Korrekt', color='green')
    plt.hist(validated_df[~validated_df['correct']]['confidence'], 
             bins=30, alpha=0.5, label='Falsch', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Anzahl Tiles')
    plt.title('Confidence-Verteilung nach Korrektheit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'), dpi=150)
    plt.show()
    
else:
    print("WARNUNG: Keine Tiles mit Referenzdaten gefunden!")
    print("Mögliche Gründe:")
    print("1. CRS stimmt nicht überein")
    print("2. Tiles und Shapefile überlappen sich nicht räumlich")
    print("3. Baumart-Spalte im Shapefile hat andere Bezeichnungen")

# ============================================
# SPEICHERE ERGEBNISSE
# ============================================
print(f"\nSpeichere Ergebnisse nach {OUTPUT_DIR}...")

# Speichere GeoDataFrame
validated_df.to_file(os.path.join(OUTPUT_DIR, 'predictions_validated.gpkg'), driver='GPKG')

# Speichere CSV für weitere Analyse
validated_df[['tile_name', 'predicted_class', 'reference_class', 'correct', 'confidence']].to_csv(
    os.path.join(OUTPUT_DIR, 'predictions_validated.csv'), index=False
)

print("\nFertig! Alle Ergebnisse gespeichert in:", OUTPUT_DIR)

# ============================================
# ZUSAMMENFASSUNG
# ============================================
print("\n" + "="*60)
print("ZUSAMMENFASSUNG TESTREGION-EVALUATION")
print("="*60)
print(f"Tiles gesamt: {len(results_df)}")
print(f"Tiles mit Referenz: {len(validated_df) if 'validated_df' in locals() else 0}")
if 'overall_accuracy' in locals():
    print(f"Gesamtgenauigkeit: {overall_accuracy:.2%}")
    print("\nKlassenweise Performance:")
    for class_name in classes:
        pred_count = (validated_df['predicted_class'] == class_name).sum()
        ref_count = (validated_df['reference_class'] == class_name).sum()
        if ref_count > 0:
            recall = ((validated_df['predicted_class'] == class_name) & 
                     (validated_df['reference_class'] == class_name)).sum() / ref_count
            print(f"  {class_name}: Recall={recall:.2%}, Predicted={pred_count}, Reference={ref_count}")
print("="*60)