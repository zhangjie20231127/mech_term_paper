from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast


def _sanitize_tensor(tensor: torch.Tensor, clamp: float = 30.0) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=clamp, neginf=-clamp)
    return tensor.clamp_(-clamp, clamp)


def _pick_group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def position(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    loc_w = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).unsqueeze(0).repeat(height, 1)
    loc_h = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).unsqueeze(1).repeat(1, width)
    return torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], dim=0).unsqueeze(0)


class ACmix(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_att: int = 7, head: int = 4, kernel_conv: int = 3, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.head_dim = out_planes // head

        self.rate1 = nn.Parameter(torch.tensor([0.5]))
        self.rate2 = nn.Parameter(torch.tensor([0.5]))
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(
            self.kernel_conv * self.kernel_conv * self.head_dim,
            out_planes,
            kernel_size=self.kernel_conv,
            bias=True,
            groups=self.head_dim,
            padding=1,
            stride=stride,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for index in range(self.kernel_conv * self.kernel_conv):
            kernel[index, index // self.kernel_conv, index % self.kernel_conv] = 1.0
        kernel = kernel.unsqueeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(kernel, requires_grad=True)
        if self.dep_conv.bias is not None:
            nn.init.zeros_(self.dep_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize_tensor(x)
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        batch_size, _, height, width = q.shape
        height_out, width_out = height // self.stride, width // self.stride

        pe = self.conv_p(position(height, width, x.device, x.dtype))
        q_att = q.view(batch_size * self.head, self.head_dim, height, width) * scaling
        k_att = k.view(batch_size * self.head, self.head_dim, height, width)
        v_att = v.view(batch_size * self.head, self.head_dim, height, width)

        q_pe = pe[:, :, :: self.stride, :: self.stride] if self.stride > 1 else pe
        if self.stride > 1:
            q_att = q_att[:, :, :: self.stride, :: self.stride]

        unfold_k = self.unfold(self.pad_att(k_att)).view(batch_size * self.head, self.head_dim, self.kernel_att * self.kernel_att, height_out, width_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, height_out, width_out)
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = _sanitize_tensor(att)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(batch_size * self.head, self.head_dim, self.kernel_att * self.kernel_att, height_out, width_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(batch_size, self.out_planes, height_out, width_out)

        fused = torch.cat(
            [
                q.view(batch_size, self.head, self.head_dim, height * width),
                k.view(batch_size, self.head, self.head_dim, height * width),
                v.view(batch_size, self.head, self.head_dim, height * width),
            ],
            dim=1,
        )
        f_all = self.fc(fused)
        f_conv = f_all.permute(0, 2, 1, 3).reshape(batch_size, -1, height, width)
        out_conv = self.dep_conv(f_conv)
        rate1 = self.rate1.float().clamp(0.0, 1.0)
        rate2 = self.rate2.float().clamp(0.0, 1.0)
        return _sanitize_tensor(rate1 * out_att + rate2 * out_conv)


class PositionAttentionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.acmix = ACmix(in_planes=d_model, out_planes=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(_sanitize_tensor(x))
        return self.acmix(y).flatten(2).permute(0, 2, 1)


class ChannelAttentionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.acmix = ACmix(in_planes=d_model, out_planes=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(_sanitize_tensor(x))
        return self.acmix(y).flatten(2)


class DModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        p_out = self.position_attention_module(x).permute(0, 2, 1).view(batch_size, channels, height, width)
        c_out = self.channel_attention_module(x).view(batch_size, channels, height, width)
        return _sanitize_tensor(p_out + c_out)


class Attention(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize_tensor(x)
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        return self.relu(_sanitize_tensor(x1 + x2 + x3))


class CausalityMapBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < 0:
            min_values, _ = torch.min(x, dim=2)
            x = x + torch.abs(min_values).unsqueeze(-1)

        maximum_values = torch.max(x, dim=2)[0]
        max_f = torch.max(maximum_values, dim=1)[0]
        x = torch.nan_to_num(x / (max_f.unsqueeze(1).unsqueeze(2) + 1e-8), nan=0.0)
        sum_values = torch.nan_to_num(torch.sum(x, dim=2), nan=0.0)
        maximum_values = torch.max(x, dim=2)[0]
        matrix = torch.einsum("bi,bj->bij", maximum_values, maximum_values)
        return torch.nan_to_num(matrix / (sum_values.unsqueeze(1) + 1e-8), nan=0.0)


class CrocodileFeatureBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        freeze_seq: bool = False,
        freeze_dnet: bool = False,
        bypass_complex_block: bool = False,
    ) -> None:
        super().__init__()
        self.bypass_complex_block = bypass_complex_block
        self.simple_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.simple_norm = nn.GroupNorm(num_groups=_pick_group_count(hidden_dim), num_channels=hidden_dim)
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(2),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dnet = DModule(d_model=in_channels, kernel_size=3)
        self.conv = nn.Conv1d(2 * in_channels, hidden_dim, kernel_size=1)
        self.att = Attention(channel=in_channels)
        self.output_norm = nn.GroupNorm(num_groups=_pick_group_count(hidden_dim), num_channels=hidden_dim)
        self.freeze_seq = freeze_seq
        self.freeze_dnet = freeze_dnet
        if self.bypass_complex_block:
            for module in (self.seq, self.dnet, self.conv, self.att, self.output_norm):
                for param in module.parameters():
                    param.requires_grad = False
        if self.freeze_seq:
            for param in self.seq.parameters():
                param.requires_grad = False
        if self.freeze_dnet:
            for param in self.dnet.parameters():
                param.requires_grad = False

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.bypass_complex_block:
            return _sanitize_tensor(self.simple_norm(self.simple_proj(_sanitize_tensor(src))))

        batch_size, _, height, width = src.shape
        with autocast(device_type=src.device.type, enabled=False):
            src = _sanitize_tensor(src)
            src0 = self.dnet(src)
            src00 = _sanitize_tensor(self.seq(src))
            src1 = _sanitize_tensor(src00.flatten(2) + self.att(src).flatten(2))
            src2 = torch.cat((src0.flatten(2), src1), dim=1)
            features = self.conv(_sanitize_tensor(src2)).reshape(batch_size, -1, height, width)
            features = self.output_norm(_sanitize_tensor(features))
        return _sanitize_tensor(features)
