import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(n_outputs, reduction=8) if use_se else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        if self.se is not None:
            out = self.se(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            k = kernel_sizes[i] if isinstance(kernel_sizes, (list, tuple)) else kernel_sizes
            layers.append(TemporalBlock(
                in_channels, out_channels, k, stride=1,
                dilation=dilation_size,
                padding=(k - 1) * dilation_size,
                dropout=dropout,
                use_se=True
            ))
        self.network = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x

    def forward_with_intermediates(self, x, extract_indices=None):
        if extract_indices is None:
            return self.forward(x), []
        features = []
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i in extract_indices:
                features.append(x)
        return x, features


class SelfAttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        scores = self.attn(x_t).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(1)
        out = torch.bmm(weights, x_t).squeeze(1)
        return out


class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes=2, kernel_sizes=3, dropout=0.2):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(num_channels)
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.fusion_levels = sorted(set([2, 4, len(num_channels) - 1]))

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_sizes, dropout)

        fusion_channels = 0
        for idx in self.fusion_levels:
            fusion_channels += num_channels[idx]
        fusion_channels *= 2

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        classifier_input = fusion_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, classifier_input // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_input // 2, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        _, features = self.tcn.forward_with_intermediates(x, extract_indices=set(self.fusion_levels))

        fused = []
        for feat in features:
            gap = self.gap(feat).squeeze(-1)
            gmp = self.gmp(feat).squeeze(-1)
            fused.append(gap)
            fused.append(gmp)

        out = torch.cat(fused, dim=-1)
        out = self.classifier(out)
        return out
