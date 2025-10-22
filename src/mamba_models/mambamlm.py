# flake8: noqa

from ..mamba_schema.slidingwindowdataset import get_position_encoding
from ..mamba_schema.mambatrainer import SNPMambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MixerModel  # type: ignore
from torch import nn
import torch
from collections import namedtuple
from typing import Literal
import random
# from timm.models.layers import trunc_normal_, lecun_normal_

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def forward_mix(self, input_ids, pos_encoding_indices, mask, inference_params=None, patching=False, shuffle=False, total_seq_len=10_000, pos_pred = False, **mixer_kwargs):

    if patching:
        input_ids = input_ids.unsqueeze(1)  # (batch_size, n_channels(1), n_labels+1, seq_len)
        hidden_states = self.conv(input_ids)  # (batch_size, embedding_size, 1, seq_len)
        hidden_states = hidden_states.squeeze(dim=2).permute(0, 2, 1)  # (batch_size, seq_len, embedding_size)

        mask = mask.unsqueeze(-1)  # (B, L, 1)
        mask_embedding = self.mask_embedding.view(1, 1, -1).expand(hidden_states.size(0), hidden_states.size(1), -1)
        hidden_states = torch.where(mask, mask_embedding.to(hidden_states.dtype), hidden_states)
    else:
        hidden_states = self.embedding(input_ids.long())

    if not pos_pred:
        pos_enc = get_position_encoding(
            sequence_pe=self.whole_position_encoding, 
            indices=pos_encoding_indices,
            shuffle=shuffle,
            total_seq_len=total_seq_len
        )
        hidden_states = hidden_states + pos_enc
    
    residual = None
    for layer in self.layers:
        hidden_states, residual = layer(
            hidden_states, residual, inference_params=inference_params, **mixer_kwargs
        )
    
    if not self.fused_add_norm:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
    else:
        hidden_states = layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=isinstance(self.norm_f, RMSNorm)
        )
    
    return hidden_states

MixerModel.forward = forward_mix


def forward_mlmh(self, input_ids, pos_encoding_indices, mask, position_ids=None, patching=False, inference_params=None, num_last_tokens=0, shuffle=False, total_seq_len=10_000, pos_pred=False, causal = False, **mixer_kwargs):
    hidden_states = self.backbone(input_ids, pos_encoding_indices, mask, inference_params=inference_params, patching=patching, shuffle=shuffle, total_seq_len=total_seq_len, pos_pred=pos_pred, **mixer_kwargs)
    
    if causal:
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
    return hidden_states

MambaLMHeadModel.forward = forward_mlmh


class BiMambaMLM(nn.Module):
    def __init__(
        self,
        config,
        whole_position_encoding: torch.Tensor,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple = (4, 1),
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        pos_pred=False
    ):
        super().__init__()

        self.pos_pred = pos_pred

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        )

        self.total_seq_len = total_seq_len

        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head
        del self.backbone.backbone.embedding

        self.backbone.backbone.conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.d_model,
            kernel_size=conv_kernel_dim,
            stride=1,
        )

        self.backbone.backbone.mask_embedding = nn.Parameter(torch.empty(config.d_model).normal_(mean=0.0, std=0.02))
        
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, pos_encoding_indices, mask, inference_params=None, num_last_tokens=0, **kwargs):
        hidden_states = self.backbone(
            input_ids=input_ids,
            pos_encoding_indices=pos_encoding_indices,
            mask=mask,
            patching=True,
            inference_params=inference_params,
            shuffle=False,
            total_seq_len=self.total_seq_len,
            pos_pred=self.pos_pred,
            **kwargs,
        )  # shape: (B, L, D)

        return self.classifier(hidden_states)

