from __future__ import annotations
import base64
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from spike.featurize import FEATURES_PER_TOKEN, NUM_SPARSE_FEATURES

HIDDEN = 32
QUANTIZATION_BITS = 6
QMAX_INT6 = (1 << (QUANTIZATION_BITS - 1)) - 1  # 31

ROLES: tuple[str, ...] = (
    "NONE",
    "FIELD",
    "VALUE",
    "OP_EQ",
    "OP_NEQ",
    "OP_GT",
    "OP_LT",
    "OP_CONTAINS",
    "OP_BEFORE",
    "OP_AFTER",
    "AND",
    "NEGATE",
)
ROLE_TO_ID: dict[str, int] = {r: i for i, r in enumerate(ROLES)}
NUM_ROLES = len(ROLES)  # 12


def fake_quantize_int6(w: torch.Tensor, bits: int = QUANTIZATION_BITS) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = w.detach().abs().max().clamp(min=1e-6) / float(qmax)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    w_deq = q * scale
    return w + (w_deq - w).detach()


def quantize_tensor_int6(w: torch.Tensor, bits: int = QUANTIZATION_BITS) -> tuple[list[int], float]:
    qmax = (1 << (bits - 1)) - 1
    w_cpu = w.detach().cpu().float()
    max_abs = float(w_cpu.abs().max().item())
    scale = max(max_abs, 1e-6) / float(qmax)
    # Round scale to float32 precision so TS and Python reconstruct identical floats
    scale = float(torch.tensor(scale, dtype=torch.float32).item())
    q = torch.clamp(torch.round(w_cpu / scale), -qmax, qmax).to(torch.int8)
    return q.reshape(-1).tolist(), scale


def pack_int6_bytes(int6_values: list[int]) -> bytes:
    """Pack signed 6-bit integers (-31..31) into a contiguous bitstream (6 bits per weight)."""
    # Offset by +32 into unsigned 0..63
    unsigned = [(v + 32) & 0x3F for v in int6_values]
    out = bytearray()
    acc = 0
    bits_in_acc = 0
    for u in unsigned:
        acc = (acc << 6) | u
        bits_in_acc += 6
        while bits_in_acc >= 8:
            bits_in_acc -= 8
            out.append((acc >> bits_in_acc) & 0xFF)
    if bits_in_acc > 0:
        out.append((acc << (8 - bits_in_acc)) & 0xFF)
    return bytes(out)


def unpack_int6_bytes(packed: bytes, count: int) -> list[int]:
    """Unpack signed 6-bit integers (-31..31) from a contiguous 6-bit bitstream."""
    out: list[int] = []
    acc = 0
    bits_in_acc = 0
    for b in packed:
        acc = (acc << 8) | b
        bits_in_acc += 8
        while bits_in_acc >= 6 and len(out) < count:
            bits_in_acc -= 6
            u = (acc >> bits_in_acc) & 0x3F
            out.append(u - 32)
    return out


class QATLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bits: int = QUANTIZATION_BITS) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_int6(self.weight, self.bits)
        b = fake_quantize_int6(self.bias, self.bits)
        return F.linear(x, w, b)


class QueryTagger(nn.Module):
    """Bidirectional gated-affine-scan sequence tagger (~29.6k parameters, 6-bit QAT).

    Architecture:
    1. Summed sparse-feature embedding (NUM_SPARSE_FEATURES=52, HIDDEN=32)
    2. 5-token depthwise conv1d + 2-step neighbor gathers (±1, ±2) -> local MLP (192 -> 64 -> 32)
    3. Forward & backward gated affine scans: state[t] = gate[t] * state[t-1] + candidate[t]
    4. Masked mean-pooled global context (32)
    5. Two-layer classification head (128 -> 64 -> 13: 12 roles + 1 clause boundary)
    """

    def __init__(
        self,
        num_features: int = NUM_SPARSE_FEATURES,
        hidden: int = HIDDEN,
        quantization_bits: int = QUANTIZATION_BITS,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.quantization_bits = quantization_bits

        self.embed_weight = nn.Parameter(torch.randn(num_features, hidden) * 0.12)
        # Padding row 0 initialized near zero
        with torch.no_grad():
            self.embed_weight[0].zero_()

        # 5-token depthwise conv1d weight [hidden, 1, 5] and bias [hidden]
        self.dw_conv_weight = nn.Parameter(torch.randn(hidden, 1, 5) * 0.15)
        self.dw_conv_bias = nn.Parameter(torch.zeros(hidden))

        # Local context mixing over [center, dw_conv, left1, right1, left2, right2] = 6 * 32 = 192
        self.local_fc1 = QATLinear(hidden * 6, hidden * 2, bits=quantization_bits)
        self.local_fc2 = QATLinear(hidden * 2, hidden, bits=quantization_bits)

        # Forward gated affine scan: state[t] = gate[t] * state[t-1] + candidate[t]
        self.fwd_gate = QATLinear(hidden, hidden, bits=quantization_bits)
        self.fwd_cand = QATLinear(hidden, hidden, bits=quantization_bits)

        # Backward gated affine scan
        self.bwd_gate = QATLinear(hidden, hidden, bits=quantization_bits)
        self.bwd_cand = QATLinear(hidden, hidden, bits=quantization_bits)

        # Two-layer head over [x_local, fwd_state, bwd_state, global_ctx] = 4 * 32 = 128
        self.head_fc1 = QATLinear(hidden * 4, hidden * 2, bits=quantization_bits)
        self.head_fc2 = QATLinear(hidden * 2, NUM_ROLES + 1, bits=quantization_bits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        feature_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_traces: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        feature_ids: [B, T, FEATURES_PER_TOKEN] long tensor
        mask: [B, T] float or bool tensor (1 for valid tokens, 0 for pad)
        """
        B, T, _ = feature_ids.shape
        if mask is None:
            mask_f = (feature_ids[:, :, 0] != 0).float()
        else:
            mask_f = mask.float()
        mask_3d = mask_f.unsqueeze(-1)  # [B, T, 1]

        # 1. Summed sparse-feature embedding with 6-bit QAT
        emb_table = fake_quantize_int6(self.embed_weight, self.quantization_bits)
        # Zero out index 0 for padding
        emb_table = emb_table.clone()
        emb_table[0] = 0.0
        x_emb = F.embedding(feature_ids, emb_table).sum(dim=-2) * mask_3d  # [B, T, H]

        # 2. 5-token depthwise conv1d + 2 neighbor gathers (±1 and ±2)
        dw_w = fake_quantize_int6(self.dw_conv_weight, self.quantization_bits)
        dw_b = fake_quantize_int6(self.dw_conv_bias, self.quantization_bits)
        x_conv = F.conv1d(
            x_emb.transpose(1, 2),
            dw_w,
            dw_b,
            padding=2,
            groups=self.hidden,
        ).transpose(1, 2) * mask_3d  # [B, T, H]

        zeros_1 = torch.zeros(B, 1, self.hidden, device=x_emb.device, dtype=x_emb.dtype)
        zeros_2 = torch.zeros(B, 2, self.hidden, device=x_emb.device, dtype=x_emb.dtype)
        left1 = torch.cat([zeros_1, x_emb[:, :-1, :]], dim=1) * mask_3d
        right1 = torch.cat([x_emb[:, 1:, :], zeros_1], dim=1) * mask_3d
        left2 = torch.cat([zeros_2, x_emb[:, :-2, :]], dim=1) * mask_3d
        right2 = torch.cat([x_emb[:, 2:, :], zeros_2], dim=1) * mask_3d

        local_cat = torch.cat([x_emb, x_conv, left1, right1, left2, right2], dim=-1)
        x_local = (x_emb + self.local_fc2(F.relu(self.local_fc1(local_cat)))) * mask_3d

        # 3. Bidirectional gated affine scan: state[t] = gate[t] * state[t-1] + candidate[t]
        fg = torch.sigmoid(self.fwd_gate(x_local)) * mask_3d
        fc = torch.tanh(self.fwd_cand(x_local)) * mask_3d

        bg = torch.sigmoid(self.bwd_gate(x_local)) * mask_3d
        bc = torch.tanh(self.bwd_cand(x_local)) * mask_3d

        fwd_states: list[torch.Tensor] = []
        state_f = torch.zeros(B, self.hidden, device=x_emb.device, dtype=x_emb.dtype)
        for t in range(T):
            state_f = fg[:, t, :] * state_f + fc[:, t, :]
            state_f = state_f * mask_3d[:, t, :]
            fwd_states.append(state_f)
        fwd_out = torch.stack(fwd_states, dim=1)  # [B, T, H]

        bwd_states: list[torch.Tensor] = [torch.empty(0)] * T
        state_b = torch.zeros(B, self.hidden, device=x_emb.device, dtype=x_emb.dtype)
        for t in range(T - 1, -1, -1):
            state_b = bg[:, t, :] * state_b + bc[:, t, :]
            state_b = state_b * mask_3d[:, t, :]
            bwd_states[t] = state_b
        bwd_out = torch.stack(bwd_states, dim=1)  # [B, T, H]

        # 4. Mean-pooled global context
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)  # [B, 1, 1]
        global_ctx = (x_local.sum(dim=1, keepdim=True) / denom).expand(B, T, self.hidden) * mask_3d

        # 5. Two-layer classification head
        combined = torch.cat([x_local, fwd_out, bwd_out, global_ctx], dim=-1)  # [B, T, 128]
        head_hidden = F.relu(self.head_fc1(combined)) * mask_3d
        logits_all = self.head_fc2(head_hidden)  # [B, T, 13]

        role_logits = logits_all[:, :, :NUM_ROLES]
        boundary_logits = logits_all[:, :, NUM_ROLES]

        out: dict[str, torch.Tensor] = {
            "role_logits": role_logits,
            "boundary_logits": boundary_logits,
        }
        if return_traces:
            out["x_local"] = x_local
            out["fwd_gate"] = fg
            out["bwd_gate"] = bg
            out["fwd_state"] = fwd_out
            out["bwd_state"] = bwd_out
            out["head_hidden"] = head_hidden
        return out

    def export_quantized_weights(self) -> dict[str, Any]:
        """Export 6-bit quantized weights into a compact <45 KB JSON-serializable structure."""
        tensor_map: dict[str, torch.Tensor] = {
            "embed_weight": self.embed_weight.clone(),
            "dw_conv_weight": self.dw_conv_weight.squeeze(1),  # [32, 5]
            "dw_conv_bias": self.dw_conv_bias,
            "local_fc1_weight": self.local_fc1.weight,
            "local_fc1_bias": self.local_fc1.bias,
            "local_fc2_weight": self.local_fc2.weight,
            "local_fc2_bias": self.local_fc2.bias,
            "fwd_gate_weight": self.fwd_gate.weight,
            "fwd_gate_bias": self.fwd_gate.bias,
            "fwd_cand_weight": self.fwd_cand.weight,
            "fwd_cand_bias": self.fwd_cand.bias,
            "bwd_gate_weight": self.bwd_gate.weight,
            "bwd_gate_bias": self.bwd_gate.bias,
            "bwd_cand_weight": self.bwd_cand.weight,
            "bwd_cand_bias": self.bwd_cand.bias,
            "head_fc1_weight": self.head_fc1.weight,
            "head_fc1_bias": self.head_fc1.bias,
            "head_fc2_weight": self.head_fc2.weight,
            "head_fc2_bias": self.head_fc2.bias,
        }
        with torch.no_grad():
            tensor_map["embed_weight"][0].zero_()

        tensors_export: dict[str, Any] = {}
        total_params = 0
        total_packed_bytes = 0

        for name, tensor in tensor_map.items():
            q_vals, scale = quantize_tensor_int6(tensor, self.quantization_bits)
            if name == "embed_weight":
                for i in range(self.hidden):
                    q_vals[i] = 0
            packed = pack_int6_bytes(q_vals)
            total_params += len(q_vals)
            total_packed_bytes += len(packed)
            tensors_export[name] = {
                "shape": list(tensor.shape),
                "scale": scale,
                "b64": base64.b64encode(packed).decode("ascii"),
            }

        return {
            "architecture": "sieve-bidirectional-gated-affine-scan-v1",
            "hidden": self.hidden,
            "num_features": self.num_features,
            "features_per_token": FEATURES_PER_TOKEN,
            "quantization_bits": self.quantization_bits,
            "roles": list(ROLES),
            "param_count": total_params,
            "packed_int6_bytes": total_packed_bytes,
            "tensors": tensors_export,
        }
