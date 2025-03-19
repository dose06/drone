# 모델/transformer.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int= 128, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        # 입력을 d_model 차원으로 임베딩
        self.embedding = nn.Linear(input_dim, d_model)
        # Transformer 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # d_model 차원을 다시 원래 input_dim 차원으로 변환 (재구성)
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim]
        """
        # 임베딩: [batch, seq, d_model]
        x = self.embedding(x)
        # Transformer는 [seq, batch, feature] 순서를 기대하므로 transpose
        x = x.transpose(0, 1)
        # 인코더 통과: [seq, batch, d_model]
        x = self.transformer_encoder(x)
        # 다시 [batch, seq, d_model]로 변환
        x = x.transpose(0, 1)
        # 최종 출력: [batch, seq, input_dim]
        x = self.fc_out(x)
        return x
