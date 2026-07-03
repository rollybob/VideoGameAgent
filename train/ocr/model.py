"""
Tiny CRNN (CNN + BiLSTM + CTC head) for single-line OCR. Pure torch (the training
container has no torchvision). Width is downsampled by 4 (two width-halving pools),
height is collapsed by adaptive pooling, so the output is a length-W/4 sequence of
class logits suitable for CTC.
"""
from __future__ import annotations

import torch.nn as nn

WIDTH_DOWNSAMPLE = 4  # keep in sync with gen_data.py --downsample


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            _block(in_ch, 64), nn.MaxPool2d(2, 2),                   # H/2,  W/2
            _block(64, 128), nn.MaxPool2d(2, 2),                     # H/4,  W/4
            _block(128, 256), _block(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # H/8,  W/4
            _block(256, 512), _block(512, 512), nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
        )
        # F18 v2: wider RNN (256->384 hidden) + an extra 512 conv block. Owner OK'd the
        # size bump now that font-generalization is proven and accuracy is the gap.
        self.rnn = nn.LSTM(512, 384, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, x):                       # x: (B, in_ch, H, W)
        f = self.cnn(x)
        # Collapse the (fixed, small) feature height to 1 by averaging over the
        # height axis. Equivalent to AdaptiveAvgPool2d((1, None)) but ONNX-exportable
        # (the legacy exporter can't trace an adaptive pool with a dynamic width).
        f = f.mean(dim=2, keepdim=True)         # (B, 512, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1)       # (B, W/4, 512)
        f, _ = self.rnn(f)                       # (B, W/4, 512)
        return self.fc(f)                        # (B, W/4, n_classes)
