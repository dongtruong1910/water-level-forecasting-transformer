import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận vị trí
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [Seq_Len, Batch_Size, D_Model] (Do model bên dưới đã xoay chiều)
        """
        # Cộng vị trí vào (Tự động broadcast cho Batch)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 num_features: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 input_window: int,
                 output_window: int):
        super(TimeSeriesTransformer, self).__init__()

        self.model_type = 'Transformer'

        # 1. Input Embedding
        self.input_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 2. Transformer Encoder         # LƯU Ý QUAN TRỌNG: Ta bỏ 'batch_first=True' để chạy được trên PyTorch cũ
        # Mặc định PyTorch mong đợi input là: [Seq_Len, Batch, Feature]
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # 3. Output Decoder
        self.decoder = nn.Linear(d_model, output_window)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src shape ban đầu: [Batch, Input_Window, Features]
        """
        # B1: Embedding
        # [Batch, 30, 6] -> [Batch, 30, 32]
        src = self.input_embedding(src)

        # B2: XOAY CHIỀU DỮ LIỆU (Permute)
        # Transformer mặc định thích [Seq, Batch, Dim] hơn là [Batch, Seq, Dim]
        # [Batch, 30, 32] -> [30, Batch, 32]
        src = src.permute(1, 0, 2)

        # B3: Positional Encoding & Transformer
        # Input vào đây phải là [Seq, Batch, Dim]
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        # Output memory vẫn là [30, Batch, 32]

        # B4: XOAY LẠI (Permute Back) & Pooling
        # [30, Batch, 32] -> [Batch, 30, 32]
        memory = memory.permute(1, 0, 2)

        # Global Average Pooling (Lấy trung bình đặc trưng của 30 ngày)
        # [Batch, 30, 32] -> [Batch, 32]
        memory_pooled = torch.mean(memory, dim=1)

        # B5: Decode
        # [Batch, 32] -> [Batch, 7]
        output = self.decoder(memory_pooled)

        # Reshape [Batch, 7, 1]
        return output.unsqueeze(-1)