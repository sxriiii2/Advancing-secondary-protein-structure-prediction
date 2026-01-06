"""
Novel 2D CNN + RNN + BiGRU Hybrid Architecture for Protein Secondary Structure Prediction
More complex than Paper 2's 2D-RNN approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

class Conv2DBlock(nn.Module):
    """
    2D Convolutional Block for extracting spatial patterns from protein sequences
    Processes sequences as 2D feature maps
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Residual2DBlock(nn.Module):
    """Residual block for 2D CNN to enable deeper networks"""
    def __init__(self, channels, kernel_size=(3, 3), padding=(1, 1)):
        super(Residual2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class RNNBlock(nn.Module):
    """
    RNN Block for capturing sequential dependencies
    Processes temporal patterns in protein sequences
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(RNNBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, features)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        out, hidden = self.rnn(x)
        
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        
        # Apply batch norm and dropout
        out = out.permute(0, 2, 1)  # (batch, features, seq_len)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (batch, seq_len, features)
        out = self.dropout(out)
        
        return out, hidden

class BiGRUBlock(nn.Module):
    """
    Bidirectional GRU Block for capturing bidirectional context
    More efficient than BiLSTM while maintaining performance
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(BiGRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, features)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        out, hidden = self.gru(x)
        
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        
        # Apply batch norm and dropout
        out = out.permute(0, 2, 1)  # (batch, features, seq_len)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (batch, seq_len, features)
        out = self.dropout(out)
        
        return out, hidden

class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for focusing on important residues"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        residual = x
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(residual + self.dropout(attn_out))
        return out

class Protein2DCNN_RNN_BiGRU(nn.Module):
    """
    Novel Hybrid Architecture: 2D CNN + RNN + BiGRU
    More complex than Paper 2's 2D-RNN variants
    
    Architecture:
    1. Input Embedding Layer
    2. 2D CNN Blocks (spatial feature extraction)
    3. RNN Block (sequential dependencies)
    4. BiGRU Block (bidirectional context)
    5. Attention Layer (focus on important residues)
    6. Fully Connected Layers (classification)
    """
    def __init__(
        self,
        vocab_size=20,  # 20 amino acids
        embed_dim=128,
        cnn_channels=[64, 128, 256],
        rnn_hidden=256,
        bigru_hidden=256,
        num_classes=3,  # H, E, C
        max_seq_len=700,
        dropout=0.3,
        use_pssm=False,  # Whether to use PSSM features
        pssm_dim=20  # PSSM dimension (20 amino acids)
    ):
        super(Protein2DCNN_RNN_BiGRU, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_pssm = use_pssm
        
        # 1. Input Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 1b. PSSM Projection (if using PSSM)
        if use_pssm:
            # Project PSSM (20 dims) to embed_dim
            self.pssm_projection = nn.Linear(pssm_dim, embed_dim)
            # Combine one-hot embedding with PSSM
            self.feature_combine = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.pssm_projection = None
            self.feature_combine = None
        
        # 2. 2D CNN Blocks for spatial feature extraction
        self.conv2d_blocks = nn.ModuleList()
        in_channels = 1  # Start with single channel
        
        for out_channels in cnn_channels:
            self.conv2d_blocks.append(Conv2DBlock(in_channels, out_channels))
            # Add residual blocks for deeper learning
            self.conv2d_blocks.append(Residual2DBlock(out_channels))
            in_channels = out_channels
        
        # Pooling after CNN - reduce spatial dimensions while keeping sequence length
        self.cnn_pool = nn.AdaptiveAvgPool2d((max_seq_len, embed_dim))
        
        # Projection layer to match embed_dim after CNN
        # After pooling, we have (channels, embed_dim) per position
        # We'll average over channels and project to embed_dim
        self.cnn_projection = nn.Linear(embed_dim, embed_dim)  # Project from pooled features
        
        # 3. RNN Block
        self.rnn = RNNBlock(
            input_size=embed_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        # 4. BiGRU Block
        self.bigru = BiGRUBlock(
            input_size=rnn_hidden,
            hidden_size=bigru_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        # 5. Attention Layer
        self.attention = AttentionLayer(
            embed_dim=bigru_hidden * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # 6. Fully Connected Layers
        fc_input_size = bigru_hidden * 2  # Bidirectional output
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths=None, pssm=None):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, seq_len) - amino acid indices
            lengths: Optional sequence lengths for variable-length sequences
            pssm: Optional PSSM tensor of shape (batch, seq_len, 20)
        Returns:
            Output tensor of shape (batch, seq_len, num_classes)
        """
        batch_size, seq_len = x.size()
        
        # 1. Embedding
        x_emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # 1b. Add PSSM features if provided
        if self.use_pssm and pssm is not None:
            # Project PSSM to same dimension
            pssm_proj = self.pssm_projection(pssm)  # (batch, seq_len, embed_dim)
            # Concatenate and combine
            x_combined = torch.cat([x_emb, pssm_proj], dim=-1)  # (batch, seq_len, embed_dim*2)
            x_emb = self.feature_combine(x_combined)  # (batch, seq_len, embed_dim)
        
        # 2. Prepare for 2D CNN: Reshape to (batch, 1, seq_len, embed_dim)
        x_2d = x_emb.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        
        # 3. Apply 2D CNN blocks
        for block in self.conv2d_blocks:
            x_2d = block(x_2d)
        
        # 4. Pool and reshape back to sequence format
        x_2d = self.cnn_pool(x_2d)  # (batch, channels, seq_len, embed_dim)
        # Reshape: (batch, channels, seq_len, embed_dim) -> (batch, seq_len, channels, embed_dim)
        x_2d = x_2d.permute(0, 2, 1, 3)  # (batch, seq_len, channels, embed_dim)
        # Average over channel dimension and project
        batch_size, seq_len, channels, embed_dim = x_2d.shape
        x_2d = x_2d.mean(dim=2)  # (batch, seq_len, embed_dim) - average over channels
        x_2d = self.cnn_projection(x_2d)  # Project to ensure correct dimension
        
        # Update lengths after CNN processing (sequence length should be preserved)
        # Don't pass lengths to RNN/BiGRU since all sequences are padded to same length
        # 5. RNN Block
        x_rnn, _ = self.rnn(x_2d, lengths=None)  # (batch, seq_len, rnn_hidden)
        
        # 6. BiGRU Block
        x_bigru, _ = self.bigru(x_rnn, lengths=None)  # (batch, seq_len, bigru_hidden*2)
        
        # 7. Attention Layer
        x_attn = self.attention(x_bigru)  # (batch, seq_len, bigru_hidden*2)
        
        # 8. Fully Connected Layers (apply to each time step)
        # Get actual dimensions from tensor
        actual_batch_size, actual_seq_len, features = x_attn.size()
        
        # Reshape for FC: (batch * seq_len, features)
        x_flat = x_attn.contiguous().view(actual_batch_size * actual_seq_len, features)
        
        # Apply FC layers
        output = self.fc_layers(x_flat)  # (batch * seq_len, num_classes)
        
        # Reshape back: (batch, seq_len, num_classes)
        output = output.view(actual_batch_size, actual_seq_len, -1)
        
        return output
    
    def predict(self, x, lengths=None):
        """Make predictions with softmax"""
        logits = self.forward(x, lengths)
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        return predictions, probs

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = Protein2DCNN_RNN_BiGRU(
        vocab_size=20,
        embed_dim=128,
        cnn_channels=[64, 128, 256],
        rnn_hidden=256,
        bigru_hidden=256,
        num_classes=3,
        max_seq_len=700,
        dropout=0.3
    )
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    
    # Test with dummy data
    batch_size = 4
    seq_len = 100
    dummy_input = torch.randint(0, 20, (batch_size, seq_len))
    
    print(f"\nTesting with input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    predictions, probs = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probs.shape}")

