"""
Training Script for 2D CNN + RNN + BiGRU Model
Includes advanced training features: learning rate scheduling, early stopping, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model_2dcnn_rnn_bigru import Protein2DCNN_RNN_BiGRU
from preprocess_data import ProteinDataset

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def calculate_metrics(y_true, y_pred, num_classes=3):
    """Calculate accuracy, precision, recall, F1-score"""
    # Flatten
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred.cpu().numpy().flatten()
    
    # Remove padding (class 0 for padding, but we use actual padding token)
    mask = y_true_flat != -1  # Assuming -1 is padding
    if mask.sum() > 0:
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }

def train_epoch(model, dataloader, criterion, optimizer, device, use_pssm=False):
    """Train for one epoch - Optimized for CPU"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # Use gradient accumulation for effective larger batch size on CPU
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
    
    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(pbar):
        # Handle with or without PSSM
        if use_pssm and len(batch_data) == 4:
            sequences, structures, lengths, pssm = batch_data
            sequences = sequences.to(device, non_blocking=False)
            structures = structures.to(device, non_blocking=False)
            pssm = pssm.to(device, non_blocking=False)
            # Forward pass with PSSM
            outputs = model(sequences, lengths, pssm=pssm)
        else:
            sequences, structures, lengths = batch_data[:3]
            sequences = sequences.to(device, non_blocking=False)
            structures = structures.to(device, non_blocking=False)
            # Forward pass without PSSM
            outputs = model(sequences, lengths)
        
        # Reshape for loss calculation
        outputs_flat = outputs.view(-1, outputs.size(-1))
        structures_flat = structures.view(-1)
        
        # Calculate loss (divide by accumulation steps)
        loss = criterion(outputs_flat, structures_flat) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up
        
        # Predictions (only for metrics, not every batch to save memory)
        if batch_idx % 10 == 0:  # Sample every 10 batches
            preds = torch.argmax(outputs_flat, dim=1)
            all_preds.append(preds.cpu())  # Move to CPU to save GPU memory
            all_targets.append(structures_flat.cpu())
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Final update if needed
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = calculate_metrics(all_targets, all_preds)
    else:
        metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    return avg_loss, metrics

def validate(model, dataloader, criterion, device, use_pssm=False):
    """Validate model - Optimized for CPU"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Validating")):
            # Handle with or without PSSM
            if use_pssm and len(batch_data) == 4:
                sequences, structures, lengths, pssm = batch_data
                sequences = sequences.to(device, non_blocking=False)
                structures = structures.to(device, non_blocking=False)
                pssm = pssm.to(device, non_blocking=False)
                outputs = model(sequences, lengths, pssm=pssm)
            else:
                sequences, structures, lengths = batch_data[:3]
                sequences = sequences.to(device, non_blocking=False)
                structures = structures.to(device, non_blocking=False)
                outputs = model(sequences, lengths)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            structures_flat = structures.view(-1)
            
            loss = criterion(outputs_flat, structures_flat)
            total_loss += loss.item()
            
            # Sample predictions to save memory
            if batch_idx % 5 == 0:  # Sample every 5 batches
                preds = torch.argmax(outputs_flat, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(structures_flat.cpu())
    
    avg_loss = total_loss / len(dataloader)
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = calculate_metrics(all_targets, all_preds)
    else:
        metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    return avg_loss, metrics

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda',
    save_dir='checkpoints',
    use_pssm=False
):
    """Main training function"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Loss and optimizer - Optimized for CPU
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # Use Adam instead of AdamW for slightly better CPU performance
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Set torch to use efficient CPU operations
    torch.set_num_threads(4)  # Use 4 threads for CPU (adjust based on your CPU cores)
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True  # Enable MKL-DNN for better CPU performance
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    
    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, use_pssm=use_pssm)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_pssm=use_pssm)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
            }, save_path / 'best_model.pt')
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.4f})")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model and history
    torch.save(model.state_dict(), save_path / 'final_model.pt')
    
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}/")
    
    return history

def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined
    axes[1, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', alpha=0.7)
    axes[1, 1].plot(epochs, history['train_f1'], 'b--', label='Train F1', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_f1'], 'r--', label='Val F1', alpha=0.7)
    axes[1, 1].set_title('All Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path / 'training_curves.png'}")

def main():
    """Main training function"""
    # Load processed data
    print("Loading processed data...")
    with open('preprocessed data/train_sequences.pkl', 'rb') as f:
        train_seqs = pickle.load(f)
    with open('preprocessed data/train_structures.pkl', 'rb') as f:
        train_structs = pickle.load(f)
    
    with open('preprocessed data/val_sequences.pkl', 'rb') as f:
        val_seqs = pickle.load(f)
    with open('preprocessed data/val_structures.pkl', 'rb') as f:
        val_structs = pickle.load(f)
    
    # Check if PSSM features exist
    use_pssm = False
    train_pssm = None
    val_pssm = None
    
    pssm_train_path = Path('preprocessed data/train_pssm.pkl')
    pssm_val_path = Path('preprocessed data/val_pssm.pkl')
    
    if pssm_train_path.exists() and pssm_val_path.exists():
        print("Loading PSSM features...")
        with open(pssm_train_path, 'rb') as f:
            train_pssm = pickle.load(f)
        with open(pssm_val_path, 'rb') as f:
            val_pssm = pickle.load(f)
        use_pssm = True
        print("[OK] PSSM features loaded! Using PSSM-enhanced model.")
    else:
        print("[INFO] No PSSM features found. Training with one-hot encoding only.")
        print("       To use PSSM: Run generate_pssm_features.py after downloading database.")
    
    # Create datasets - Match model's max_seq_len
    train_dataset = ProteinDataset(train_seqs, train_structs, max_len=500, pssm_features=train_pssm)
    val_dataset = ProteinDataset(val_seqs, val_structs, max_len=500, pssm_features=val_pssm)
    
    # OPTIMIZED FOR THINKPAD T480 (CPU-only, limited RAM)
    # Reduced batch size and model complexity for efficient CPU training
    batch_size = 4  # Reduced from 16 for CPU/memory efficiency
    num_workers = 0  # Set to 0 for Windows compatibility and to avoid memory issues
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for CPU
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )
    
    # Create model - OPTIMIZED for CPU training
    device = torch.device('cpu')  # Force CPU for ThinkPad T480
    print(f"Using device: {device}")
    print("Optimized for ThinkPad T480 (CPU-only training)")
    
    # Reduced model complexity for CPU efficiency
    model = Protein2DCNN_RNN_BiGRU(
        vocab_size=20,
        embed_dim=64,  # Reduced from 128
        cnn_channels=[32, 64, 128],  # Reduced from [64, 128, 256]
        rnn_hidden=128,  # Reduced from 256
        bigru_hidden=128,  # Reduced from 256
        num_classes=3,
        max_seq_len=500,  # Reduced from 700 to save memory
        dropout=0.3,
        use_pssm=use_pssm  # Enable PSSM if available
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train with optimized settings
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,  # Reduced from 50 for faster iteration
        learning_rate=0.001,
        device=device,
        save_dir='checkpoints',
        use_pssm=use_pssm  # Pass PSSM flag
    )
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()

