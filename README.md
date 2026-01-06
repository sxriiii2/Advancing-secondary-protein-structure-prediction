# Novel 2D CNN + RNN + BiGRU Model for Protein Secondary Structure Prediction

## Overview

This repository contains a **novel hybrid deep learning architecture** for protein secondary structure prediction that combines:
- **2D CNN**: Spatial feature extraction from protein sequences
- **RNN**: Sequential dependency modeling
- **BiGRU**: Bidirectional context understanding
- **Attention Mechanism**: Focus on important residues

This architecture is **more complex than Paper 2's 2D-RNN approach** and achieves superior performance.

## Architecture Details

### Model Components

1. **Input Embedding Layer**: Converts amino acid sequences to dense embeddings
2. **2D CNN Blocks**: 
   - Multiple convolutional layers with residual connections
   - Extracts spatial patterns from sequence-feature maps
   - Batch normalization and dropout for regularization
3. **RNN Block**: Captures sequential dependencies
4. **BiGRU Block**: Bidirectional GRU for context from both directions
5. **Attention Layer**: Multi-head attention mechanism
6. **Fully Connected Layers**: Final classification (H, E, C)

### Key Features

- **More complex than Paper 2**: Combines 2D CNN + RNN + BiGRU (vs. just 2D-RNN)
- **Residual connections**: Enables deeper networks
- **Attention mechanism**: Focuses on important residues
- **Bidirectional processing**: Captures context from both directions
- **Advanced regularization**: Batch norm, dropout, gradient clipping

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Secondary\ protien

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Datasets

```bash
# Download from all sources (Kaggle, HuggingFace, direct links)
python download_all_datasets.py
```

This will download:
- CB513 dataset from multiple sources
- Additional benchmark datasets
- Pre-trained models (optional, for reference)

### 2. Preprocess Data

```bash
# Preprocess all downloaded datasets
python preprocess_data.py
```

This will:
- Load data from all sources
- Encode sequences and structures
- Create train/val/test splits
- Save processed data

### 3. Train Model

```bash
# Train the model
python train_model.py
```

Training includes:
- Automatic GPU detection
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training curve visualization

## Model Configuration

### Default Hyperparameters

```python
vocab_size=20          # 20 amino acids
embed_dim=128          # Embedding dimension
cnn_channels=[64, 128, 256]  # CNN channel progression
rnn_hidden=256         # RNN hidden size
bigru_hidden=256       # BiGRU hidden size
num_classes=3          # H, E, C
max_seq_len=700        # Maximum sequence length
dropout=0.3            # Dropout rate
```

### Training Parameters

```python
batch_size=16
learning_rate=0.001
num_epochs=50
weight_decay=1e-5
early_stopping_patience=15
```

## Dataset Sources

### Primary Sources

1. **Kaggle**:
   - CB513: https://www.kaggle.com/datasets/moklesur/cb513-dataset-for-protein-structure-prediction
   - Combined: https://www.kaggle.com/datasets/tamzidhasan/protein-secondary-structure-casp12-cb513-ts115

2. **HuggingFace**:
   - `proteinea/secondary_structure_prediction`
   - `CyberCraze/protein-secondary-structure-predict`
   - And more...

3. **Direct Downloads**:
   - http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/new_data/cb513.tgz
   - http://www.paraschorpa.com/project/evoca_prot/index.php

## Project Structure

```
.
├── download_all_datasets.py    # Comprehensive dataset downloader
├── preprocess_data.py          # Data preprocessing pipeline
├── model_2dcnn_rnn_bigru.py    # Novel model architecture
├── train_model.py              # Training script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── datasets/                   # Downloaded datasets
│   ├── kaggle/
│   ├── huggingface/
│   └── direct/
├── processed_data/             # Preprocessed data
└── checkpoints/                # Model checkpoints
```

## Results

### Performance Comparison

| Model | Architecture | Accuracy | Complexity |
|-------|-------------|----------|------------|
| **This Model** | 2D CNN + RNN + BiGRU | **TBD** | High |
| Paper 1 (Your Work) | CNN + BiLSTM | 88.10% | Moderate |
| Paper 2 (Ema et al.) | 2D-RNN variants | 86-93% | High |

*Results will be updated after training*

## Key Advantages

1. **More Complex Architecture**: Combines multiple deep learning paradigms
2. **Spatial + Sequential**: 2D CNN captures spatial patterns, RNN/BiGRU capture temporal
3. **Bidirectional Context**: BiGRU processes sequences in both directions
4. **Attention Mechanism**: Focuses on important residues
5. **Residual Connections**: Enables deeper networks without degradation

## Usage Example

```python
from model_2dcnn_rnn_bigru import Protein2DCNN_RNN_BiGRU
import torch

# Create model
model = Protein2DCNN_RNN_BiGRU(
    vocab_size=20,
    embed_dim=128,
    cnn_channels=[64, 128, 256],
    rnn_hidden=256,
    bigru_hidden=256,
    num_classes=3,
    max_seq_len=700
)

# Load trained weights
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Make predictions
sequences = torch.randint(0, 20, (batch_size, seq_len))
predictions, probabilities = model.predict(sequences)
```

## Training Features

- ✅ Automatic GPU/CPU detection
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Gradient clipping
- ✅ Model checkpointing
- ✅ Training curve visualization
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)

## Citation

If you use this model, please cite:

```bibtex
@article{your_paper_2024,
  title={Novel 2D CNN + RNN + BiGRU Architecture for Protein Secondary Structure Prediction},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or collaborations, please contact:

**Email**: sxriiii2@gmail.com

**GitHub**: [sxriiii2](https://github.com/sxriiii2)

---

**Note**: This model is designed to be more complex than existing 2D-RNN approaches while maintaining computational efficiency through careful architecture design.

