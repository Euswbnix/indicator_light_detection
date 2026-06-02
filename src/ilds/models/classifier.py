"""Stage2 分类器：MobileViT-XXS（自包含实现，仅依赖 torch）。

为什么自己实现：环境无 timm 且无法联网安装。MobileViT 体量小，从头实现可控。
CNN(局部纹理：图标形状) + Transformer(全局特征：颜色分布) 混合，正好契合
“同形不同色”需要全局颜色信息的需求。

输入 128x128x3，输出 NUM_CLASSES（含 not_a_light）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_silu(inp, oup, k=3, s=1, g=1):
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


class InvertedResidual(nn.Module):
    """MobileNetV2 MBConv。"""
    def __init__(self, inp, oup, stride, expand):
        super().__init__()
        self.use_res = stride == 1 and inp == oup
        hidden = int(round(inp * expand))
        layers = []
        if expand != 1:
            layers.append(conv_bn_silu(inp, hidden, k=1))
        layers += [
            conv_bn_silu(hidden, hidden, k=3, s=stride, g=hidden),  # depthwise
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileViTBlock(nn.Module):
    """局部表示(conv) -> 展开成 patch -> Transformer 全局建模 -> 折回 -> 融合。"""
    def __init__(self, dim, depth, channel, kernel=3, patch=2, mlp_ratio=2.0, heads=4):
        super().__init__()
        self.ph = self.pw = patch
        self.local_rep = nn.Sequential(
            conv_bn_silu(channel, channel, k=kernel),
            nn.Conv2d(channel, dim, 1, 1, 0, bias=False),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.proj = nn.Conv2d(dim, channel, 1, 1, 0, bias=False)
        self.fusion = conv_bn_silu(2 * channel, channel, k=kernel)

    def forward(self, x):
        y = x.clone()
        x = self.local_rep(x)
        B, d, H, W = x.shape
        ph, pw = self.ph, self.pw
        # pad 到 patch 整数倍
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        nh, nw = H // ph, W // pw
        # (B,d,H,W) -> (B*ph*pw, nh*nw, d)
        x = x.reshape(B, d, nh, ph, nw, pw).permute(0, 3, 5, 2, 4, 1)
        x = x.reshape(B * ph * pw, nh * nw, d)
        x = self.transformer(x)
        x = x.reshape(B, ph, pw, nh, nw, d).permute(0, 5, 3, 1, 4, 2)
        x = x.reshape(B, d, nh * ph, nw * pw)
        if pad_h or pad_w:
            x = x[:, :, :H - pad_h, :W - pad_w]
        x = self.proj(x)
        return self.fusion(torch.cat([y, x], dim=1))


class MobileViTXXS(nn.Module):
    """MobileViT-XXS，约 1.3M 参数。"""
    def __init__(self, num_classes, in_ch=3):
        super().__init__()
        # XXS 通道配置
        c = [16, 16, 24, 48, 64, 80]
        d = [64, 80, 96]            # transformer 维度
        self.stem = conv_bn_silu(in_ch, c[0], k=3, s=2)               # 1/2
        self.l1 = InvertedResidual(c[0], c[1], 1, 2)
        self.l2 = nn.Sequential(                                      # 1/4
            InvertedResidual(c[1], c[2], 2, 2),
            InvertedResidual(c[2], c[2], 1, 2),
            InvertedResidual(c[2], c[2], 1, 2),
        )
        self.l3 = nn.Sequential(                                      # 1/8
            InvertedResidual(c[2], c[3], 2, 2),
            MobileViTBlock(d[0], depth=2, channel=c[3]),
        )
        self.l4 = nn.Sequential(                                      # 1/16
            InvertedResidual(c[3], c[4], 2, 2),
            MobileViTBlock(d[1], depth=4, channel=c[4]),
        )
        self.l5 = nn.Sequential(                                      # 1/32
            InvertedResidual(c[4], c[5], 2, 2),
            MobileViTBlock(d[2], depth=3, channel=c[5]),
        )
        self.head_conv = conv_bn_silu(c[5], 320, k=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class MobileViTTwoHead(nn.Module):
    """最终版：共享骨干 + 形状头 + 颜色头。

    形状头预测形状族，颜色头预测粗颜色；推理时 (形状,颜色) 查表 -> kb_id。
    解决“同形不同色”：形状/颜色解耦，未见色变体也可能查表识别，未见组合转人工。
    注：共享骨干 + 单一输入，故增强用 受限色相(±10°)，红/黄/绿相距~120°不会混，
        既保形状的轻度色鲁棒，又不破坏颜色头。完全色不变需两独立网络（留待 phase2 调优）。
    """
    def __init__(self, num_shape, num_color, in_ch=3):
        super().__init__()
        base = MobileViTXXS(num_shape, in_ch=in_ch)   # 复用骨干
        self.backbone = nn.Sequential(
            base.stem, base.l1, base.l2, base.l3, base.l4, base.l5,
            base.head_conv, base.pool, nn.Flatten(1), base.dropout)
        self.shape_head = nn.Linear(320, num_shape)
        self.color_head = nn.Linear(320, num_color)

    def forward(self, x):
        f = self.backbone(x)
        return self.shape_head(f), self.color_head(f)


def build_classifier(num_classes):
    """baseline：单头分类器（直接出 kb 类）。"""
    return MobileViTXXS(num_classes)


def build_twohead(num_shape, num_color):
    """最终版：形状+颜色双头。"""
    return MobileViTTwoHead(num_shape, num_color)


if __name__ == "__main__":
    from ..taxonomy import NUM_CLASSES
    m = build_classifier(NUM_CLASSES)
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 3, 128, 128)
    y = m(x)
    print(f"MobileViT-XXS 参数量: {n/1e6:.2f}M | 输出: {tuple(y.shape)} (NUM_CLASSES={NUM_CLASSES})")
