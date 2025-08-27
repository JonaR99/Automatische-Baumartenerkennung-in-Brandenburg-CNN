# Verbessertes CNN-Modell für Baumartenerkennung
# Mit Data Augmentation, Learning Rate Scheduling und automatischen Plots
# OHNE Early Stopping, mit 15.000 Bildern pro Baumart

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToTensor, Resize
import rasterio
from PIL import Image

# Erstelle Ordner für Ergebnisse
results_dir = f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_dir, exist_ok=True)

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# HYPERPARAMETER - OPTIMIERT
# ============================================
BATCH_SIZE = 32  # Reduziert von 128 für bessere Gradienten
VALIDATION_SPLIT = 0.2  # 20% für Validation
EPOCHS = 50  # Vollständige 50 Epochen ohne Early Stopping
LEARNING_RATE = 0.0001  # Erhöht von 0.0001 für schnellere initiale Konvergenz
PATIENCE_LR = 8  # Für Learning Rate Scheduler
DROPOUT_RATE = 0.2  # Reduziert von 0.4
SAMPLE_SIZE = 15000  # 15.000 Bilder pro Klasse

# Seed für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# DATEN VORBEREITUNG (mit Sampling)
# ============================================
BASE_PATH = 'D:\\BA Jona\\Pringles'
os.chdir('D:\\BA Jona')

image = []
labels = []
classes = ['birke', 'buche', 'eiche', 'kiefer', 'fichte']

# Sampling für kontrolliertere Datenmenge
print("Loading dataset...")
for class_name in classes:
    class_path = os.path.join(BASE_PATH, class_name)
    all_images = [f for f in os.listdir(class_path) if f != 'annotations']
    
    # Zufälliges Sampling von 20.000 Bildern pro Klasse
    sampled_images = np.random.choice(all_images, 
                                     min(SAMPLE_SIZE, len(all_images)), 
                                     replace=False)
    
    for img_name in sampled_images:
        image.append(img_name)
        labels.append(class_name)
    
    print(f"  {class_name}: {len(sampled_images)} images loaded")

data = pd.DataFrame({'Images': image, 'labels': labels})
print(f"\nTotal dataset size: {len(data)} images")
print(f"Class distribution:\n{data['labels'].value_counts()}")

# Label Encoding
lb = LabelEncoder()
data['encoded_labels'] = lb.fit_transform(data['labels'])

# ============================================
# VERBESSERTE DATA AUGMENTATION
# ============================================
transform_train = transforms.Compose([
    transforms.RandomRotation(90),  # Rotation wichtig für Luftbilder
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ## transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Leichte Verschiebung ## War zu viel
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============================================
# DATASET KLASSE
# ============================================
class ImprovedDataset(Dataset):
    def __init__(self, img_data, img_path, transform=None, augment=False):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data.reset_index(drop=True)
        self.augment = augment
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,
                               self.img_data.loc[index, 'labels'],
                               self.img_data.loc[index, 'Images'])
        
        # Lade Bild
        image = rasterio.open(img_name)
        image = image.read()
        image = image[:3, :, :]  # RGB Kanäle
        image = ToTensor()(image)
        image = torch.permute(image, (1, 2, 0))
        
        # Resize auf 100x100
        resize_transform = Resize((100, 100))
        image = resize_transform(image)
        
        # Augmentation nur für Training
        if self.transform is not None:
            image = self.transform(image)
            
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'], dtype=torch.long)
        return image, label

# ============================================
# DATA SPLIT UND LOADER
# ============================================
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SPLIT * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

print(f"\nData split:")
print(f"  Training samples: {len(train_indices)}")
print(f"  Validation samples: {len(val_indices)}")

# Create data loaders
train_dataset = ImprovedDataset(data.iloc[train_indices], BASE_PATH, 
                                transform=transform_train, augment=True)
val_dataset = ImprovedDataset(data.iloc[val_indices], BASE_PATH, 
                              transform=transform_val, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================
# EINFACHERES MODELL (weniger Parameter)
# ============================================
# class SimplerCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(SimplerCNN, self).__init__()
        
#         # Weniger Filter!
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d(0.2)  # Weniger Dropout
        
#         # Einfacherer Classifier
#         self.fc1 = nn.Linear(64 * 12 * 12, 256)  # Kleinere FC Layer
#         self.fc2 = nn.Linear(256, num_classes)
#         self.dropout_fc = nn.Dropout(0.2)
        
#     def forward(self, x):
#         # Conv Block 1
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
#         # Conv Block 2
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout(x)
        
#         # Conv Block 3
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
#         # Flatten
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)
    
# ============================================
# VERBESSERTES MODELL
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
# TRAINING SETUP
# ============================================
model = ImprovedCNN().to(device)
print(f"\nModel initialized")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.NLLLoss()  # Für log_softmax output
optimizer = optim.Adam(model.parameters(), 
                       lr=LEARNING_RATE,
                       weight_decay=1e-5)  # Regularisierung
def warmup_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                             patience=PATIENCE_LR, verbose=True)

# ============================================
# TRAINING HISTORY TRACKING
# ============================================
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

# ============================================
# TRAINING FUNKTIONEN
# ============================================
def train_epoch(model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in tqdm(loader, desc='Training'):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch < 5:
            warmup_scheduler.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_targets

# ============================================
# PLOT FUNKTIONEN
# ============================================
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss Plot
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate Plot
    axes[2].plot(history['lr'], color='green', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    ##plt.show()
    print(f"Training history plot saved to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Prozentuale Werte hinzufügen
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.7, f'{cm_normalized[i,j]:.1%}',
                    ha='center', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    ##plt.show()
    print(f"Confusion matrix saved to: {save_path}")

def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    # Erstelle auch DataFrame für bessere Visualisierung
    report_dict = classification_report(y_true, y_pred, target_names=class_names, 
                                       output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(save_path.replace('.txt', '.csv'))
    
    return df

# ============================================
# TRAINING LOOP (OHNE EARLY STOPPING)
# ============================================
print("\n" + "="*50)
print("Starting Training")
print("="*50)
print(f"Training for {EPOCHS} epochs (no early stopping)")
print(f"Dataset: {SAMPLE_SIZE} images per class")
print("="*50)

best_val_acc = 0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 30)
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Validation
    val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, 
                                                                criterion, device)
    
    # Learning Rate Scheduling
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    
    # History tracking
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Speichere bestes Modell basierend auf Validation Accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        os.makedirs(results_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(results_dir, 'best_model.pt'))
        print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Zwischenergebnisse speichern (alle 10 Epochen)
    if epoch % 10 == 0:
        plot_training_history(history, os.path.join(results_dir, f'history_epoch_{epoch}.png'))
        print(f"Intermediate results saved at epoch {epoch}")

# ============================================
# FINALE EVALUATION
# ============================================
print("\n" + "="*50)
print("Training Completed - Final Evaluation")
print("="*50)

# Lade bestes Modell für finale Evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")

# Finale Validation
model.eval()
_, final_acc, final_preds, final_targets = validate_epoch(model, val_loader, 
                                                          criterion, device)

print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")
print(f"Best Validation Accuracy during training: {best_val_acc:.2f}%")

# ============================================
# VISUALISIERUNGEN SPEICHERN
# ============================================
print("\n" + "="*50)
print("Generating and saving visualizations...")
print("="*50)

# 1. Training History
os.makedirs(results_dir, exist_ok=True)
plot_training_history(history, os.path.join(results_dir, 'training_history.png'))

# 2. Confusion Matrix
os.makedirs(results_dir, exist_ok=True)
plot_confusion_matrix(final_targets, final_preds, classes, 
                     os.path.join(results_dir, 'confusion_matrix.png'))

# 3. Classification Report
os.makedirs(results_dir, exist_ok=True)
report_df = save_classification_report(final_targets, final_preds, classes,
                                       os.path.join(results_dir, 'classification_report.txt'))

# 4. Per-Class Accuracy Bar Plot
plt.figure(figsize=(10, 6))
class_accs = []
for i, class_name in enumerate(classes):
    class_mask = np.array(final_targets) == i
    if class_mask.sum() > 0:  # Nur wenn Samples vorhanden
        class_acc = np.mean(np.array(final_preds)[class_mask] == i) * 100
    else:
        class_acc = 0
    class_accs.append(class_acc)

colors = ['#8B4513', '#D2691E', '#CD853F', '#228B22', '#B22222']  # Farben passend zu Baumarten
bars = plt.bar(classes, class_accs, color=colors)
plt.xlabel('Baumart', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Klassenspezifische Genauigkeit', fontsize=14)
plt.ylim(0, 100)

# Werte über den Balken anzeigen
for bar, acc in zip(bars, class_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'class_accuracy.png'), dpi=150)
##plt.show()
print(f"Class accuracy plot saved")

# 5. Loss/Accuracy Entwicklung als Tabelle
os.makedirs(results_dir, exist_ok=True)
summary_df = pd.DataFrame({
    'Epoch': range(1, len(history['train_loss']) + 1),
    'Train_Loss': history['train_loss'],
    'Train_Acc': history['train_acc'],
    'Val_Loss': history['val_loss'],
    'Val_Acc': history['val_acc'],
    'Learning_Rate': history['lr']
})
summary_df.to_csv(os.path.join(results_dir, 'training_summary.csv'), index=False)
print(f"Training summary saved as CSV")

# 6. Speichere Trainingsparameter
os.makedirs(results_dir, exist_ok=True)
params = {
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(history['train_loss']),
    'initial_lr': LEARNING_RATE,
    'dropout_rate': DROPOUT_RATE,
    'sample_size_per_class': SAMPLE_SIZE,
    'total_images': len(data),
    'train_samples': len(train_indices),
    'val_samples': len(val_indices),
    'final_train_acc': history['train_acc'][-1],
    'final_val_acc': history['val_acc'][-1],
    'best_val_acc': best_val_acc,
    'final_train_loss': history['train_loss'][-1],
    'final_val_loss': history['val_loss'][-1]
}

with open(os.path.join(results_dir, 'training_params.txt'), 'w') as f:
    f.write("="*50 + "\n")
    f.write("TRAINING PARAMETERS AND RESULTS\n")
    f.write("="*50 + "\n\n")
    for key, value in params.items():
        if isinstance(value, float):
            f.write(f"{key}: {value:.4f}\n")
        else:
            f.write(f"{key}: {value}\n")

print(f"\nAll results saved to: {results_dir}")
print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)