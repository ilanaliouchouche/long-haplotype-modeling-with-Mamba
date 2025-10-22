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


def forward_mix(self, input_ids, pos_encoding_indices=None, inference_params=None, patching=False, shuffle=False, total_seq_len=10_000, pos_pred = False, **mixer_kwargs):

    if patching:
        input_ids = input_ids.unsqueeze(1)  # (batch_size, n_channels(1), n_labels+1, seq_len)
        hidden_states = self.conv(input_ids)  # (batch_size, embedding_size, 1, seq_len)
        hidden_states = hidden_states.squeeze(dim=2).permute(0, 2, 1)  # (batch_size, seq_len, embedding_size)
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


def forward_mlmh(self, input_ids, pos_encoding_indices=None, position_ids=None, patching=False, inference_params=None, num_last_tokens=0, shuffle=False, total_seq_len=10_000,
                 pos_pred=False, causal = False, **mixer_kwargs):
    hidden_states = self.backbone(input_ids, pos_encoding_indices, inference_params=inference_params, patching=patching, shuffle=shuffle, total_seq_len=total_seq_len, 
                                  pos_pred=pos_pred, **mixer_kwargs)
    
    if causal:
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
    return hidden_states

MambaLMHeadModel.forward = forward_mlmh


def gram_schmidt_orthogonalize(v: torch.Tensor) -> torch.Tensor:

    u = torch.randn_like(v)
    projection = (torch.dot(u, v) / torch.dot(v, v)) * v
    u_orthogonal = u - projection
    u_orthogonal = nn.functional.normalize(u_orthogonal, dim=0)

    return u_orthogonal

class MambaLHeadModelWithPEClassifier(nn.Module):  
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        pooling_mode: Literal["mean", "max", "cls"],
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        assert pooling_mode in ["mean", "max", "cls"], "Pooling mode must be 'mean', 'max', or 'cls'"
        self.pooling_mode = pooling_mode

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        del self.backbone.lm_head

        self.shuffle = shuffle
        self.total_seq_len = total_seq_len

        self.backbone.backbone.whole_position_encoding = whole_position_encoding
    
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        trainable_embeddings: bool = True,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> "MambaLHeadModelWithPETokenClassifier":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            trainable_embeddings=trainable_embeddings,
            embedding_init=embedding_init,
            embedding_init_mode=embedding_init_mode,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model

    def forward(self, input_ids, pos_encoding_indices, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):

        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )
        if self.pooling_mode == "mean":
            pooled_output = torch.mean(hidden_states, dim=1)
        elif self.pooling_mode == "max":
            pooled_output, _ = torch.max(hidden_states, dim=1)
        elif self.pooling_mode == "cls":
            pooled_output = hidden_states[:, 0, :]

        return self.classifier(pooled_output)  # (batch_size, num_labels)
    

class MambaLHeadModelWithPETokenClassifier(nn.Module):  

    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        trainable_embeddings: bool = True,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        self.shuffle = shuffle
        self.total_seq_len = total_seq_len

        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head
    
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        if not trainable_embeddings:
            for param in self.backbone.backbone.embedding.parameters():
                param.requires_grad = False
        if embedding_init:
            self.backbone.backbone.embedding.weight.data[0] = torch.normal(mean=0, std=0.02, size=(config.d_model,))
            self.backbone.backbone.embedding.weight.data[1] = (-self.backbone.backbone.embedding.weight.data[0]
                                                               if embedding_init_mode == "anticolinear" else  
                                                               gram_schmidt_orthogonalize(self.backbone.backbone.embedding.weight.data[0]))
        
    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        trainable_embeddings: bool = True,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> "MambaLHeadModelWithPETokenClassifier":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            trainable_embeddings=trainable_embeddings,
            embedding_init=embedding_init,
            embedding_init_mode=embedding_init_mode,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model

    def forward(self, input_ids, pos_encoding_indices, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )
        
        return self.classifier(hidden_states)
    

class MambaLHeadModelCausalLM(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        trainable_embeddings: bool = True,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        self.shuffle = shuffle
        self.total_seq_len = total_seq_len

        self.backbone.backbone.whole_position_encoding = whole_position_encoding
    
        if not trainable_embeddings:
            for param in self.backbone.backbone.embedding.parameters():
                param.requires_grad = False
        if embedding_init:
            self.backbone.backbone.embedding.weight.data[0] = torch.normal(mean=0, std=0.02, size=(config.d_model,))
            self.backbone.backbone.embedding.weight.data[1] = (-self.backbone.backbone.embedding.weight.data[0]
                                                               if embedding_init_mode == "anticolinear" else  
                                                               gram_schmidt_orthogonalize(self.backbone.backbone.embedding.weight.data[0]))

    def forward(self, input_ids, pos_encoding_indices=None, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        logits = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            pos_pred = True,
            causal=True,
            **mixer_kwargs,
        )
        
        return logits
    

class MambaLHeadModelWithPEFeatureExtractor(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        self.shuffle = shuffle
        self.total_seq_len = total_seq_len
        self.backbone.backbone.whole_position_encoding = whole_position_encoding

        del self.backbone.lm_head

    def forward(self, input_ids, pos_encoding_indices, **mixer_kwargs):
        return self.backbone(
            input_ids,
            pos_encoding_indices,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        device=None,
        dtype=None,
    ) -> "MambaLHeadModelWithPEFeatureExtractor":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            device=device,
            dtype=dtype,
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}
        model.load_state_dict(pretrained_state_dict, strict=False)
        return model

    @classmethod
    def from_tokenclassification(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> "MambaLHeadModelWithPEFeatureExtractor":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(pretrained_state_dict, strict=False)
        return model
    

class MambaLHeadModelWithPEPatch(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple= (4,1),
        pos_pred=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )

        self.shuffle = shuffle
        self.total_seq_len = total_seq_len
        self.pos_pred = pos_pred


        # self.backbone.backbone.whole_position_encoding = whole_position_encoding
        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head
        del self.backbone.backbone.embedding

        self.backbone.backbone.conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.d_model,
            kernel_size=conv_kernel_dim,
            stride=1,  # we can define the LAI resolution here
        )

        # lecun_normal_(self.backbone.backbone.conv.weight)

        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, pos_encoding_indices, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            patching=True,
            pos_pred=self.pos_pred,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )
        
        return self.classifier(hidden_states)
    
    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple= (4,1),
        device=None,
        dtype=None,
    ) -> "MambaLHeadModelWithPETokenClassifier":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            initializer_cfg=initializer_cfg,
            conv_kernel_dim=conv_kernel_dim,
            device=device,
            dtype=dtype,
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model


class BiMambaLHeadModelWithPEPatch(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple= (4,1),
        pos_pred=False,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        ) 
        self.pos_pred = pos_pred
        self.shuffle = shuffle
        self.total_seq_len = total_seq_len
        self.pos_pred = pos_pred


        # self.backbone.backbone.whole_position_encoding = whole_position_encoding
        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head
        del self.backbone.backbone.embedding

        self.backbone.backbone.conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.d_model,
            kernel_size=conv_kernel_dim,
            stride=1,  # we can define the LAI resolution here
        )

        # lecun_normal_(self.backbone.backbone.conv.weight)

        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, pos_encoding_indices, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            patching=True,
            pos_pred=self.pos_pred,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )

        return self.classifier(hidden_states)
    
    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple= (4,1),
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        pos_pred=False
    ) -> "MambaLHeadModelWithPETokenClassifier":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            initializer_cfg=initializer_cfg,
            conv_kernel_dim=conv_kernel_dim,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out,
            pos_pred=pos_pred
        )
        # pretrained_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k and "lm_head" not in k}
        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k and "whole_position_encoding" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model

class BiMambaWithPEPatchFeatureExtractor(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
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

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out,
        )

        self.pos_pred = pos_pred
        self.shuffle = shuffle
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

    def forward(self, input_ids, pos_encoding_indices, **mixer_kwargs):
        return self.backbone(
            input_ids=input_ids,
            pos_encoding_indices=pos_encoding_indices,
            patching=True,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            pos_pred=self.pos_pred,
            **mixer_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        conv_kernel_dim: tuple = (4, 1),
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        pos_pred=False
    ) -> "BiMambaWithPEPatchFeatureExtractor":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            initializer_cfg=initializer_cfg,
            conv_kernel_dim=conv_kernel_dim,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out,
            pos_pred=pos_pred
        )

        pretrained_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k and "whole_position_encoding" not in k and "lm_head" not in k}
        model.load_state_dict(pretrained_state_dict, strict=False)
        return model


class BiMambaLHeadModelWithPE(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        pos_pred=False,
        device=None,
        dtype=None,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        bimamba_type="none",
        if_divide_out=False
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        )
        self.pos_pred = pos_pred
        self.shuffle = shuffle
        self.total_seq_len = total_seq_len
        self.pos_pred = pos_pred


        # self.backbone.backbone.whole_position_encoding = whole_position_encoding
        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head

        if embedding_init:
            self.backbone.backbone.embedding.weight.data[0] = torch.normal(mean=0, std=0.02, size=(config.d_model,))
            self.backbone.backbone.embedding.weight.data[1] = (-self.backbone.backbone.embedding.weight.data[0]
                                                               if embedding_init_mode == "anticolinear" else  
                                                               gram_schmidt_orthogonalize(self.backbone.backbone.embedding.weight.data[0]))
        

        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids, pos_encoding_indices=None, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            patching=False,
            pos_pred=self.pos_pred,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )

        return self.classifier(hidden_states)
    
    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        pos_pred=False,
        device=None,
        dtype=None,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        bimamba_type="none",
        if_divide_out=False
    ) -> "BiMambaLHeadModelWithPE":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            embedding_init=embedding_init,
            embedding_init_mode=embedding_init_mode,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            pos_pred=pos_pred,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        )
        # pretrained_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k and "lm_head" not in k}
        pretrained_state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k and "whole_position_encoding" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model

class BiMambaLHeadModelWithPEFeatureExtractor(nn.Module):
    def __init__(
        self,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        pos_pred=False,
        device=None,
        dtype=None,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        bimamba_type="none",
        if_divide_out=False
    ) -> None:
        super().__init__()

        self.backbone = MambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        )
        self.pos_pred = pos_pred
        self.shuffle = shuffle
        self.total_seq_len = total_seq_len
        self.pos_pred = pos_pred


        # self.backbone.backbone.whole_position_encoding = whole_position_encoding
        self.backbone.backbone.register_buffer("whole_position_encoding", whole_position_encoding)
        del self.backbone.lm_head

        if embedding_init:
            self.backbone.backbone.embedding.weight.data[0] = torch.normal(mean=0, std=0.02, size=(config.d_model,))
            self.backbone.backbone.embedding.weight.data[1] = (-self.backbone.backbone.embedding.weight.data[0]
                                                               if embedding_init_mode == "anticolinear" else  
                                                               gram_schmidt_orthogonalize(self.backbone.backbone.embedding.weight.data[0]))
        

    def forward(self, input_ids, pos_encoding_indices=None, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(
            input_ids,
            pos_encoding_indices,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            patching=False,
            pos_pred=self.pos_pred,
            shuffle=self.shuffle,
            total_seq_len=self.total_seq_len,
            **mixer_kwargs,
        )

        return hidden_states
    
    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict,
        config: SNPMambaConfig,
        whole_position_encoding: torch.Tensor,
        shuffle: bool = False,
        total_seq_len: int = 10_000,
        initializer_cfg=None,
        pos_pred=False,
        device=None,
        dtype=None,
        embedding_init: bool = False,
        embedding_init_mode: Literal["anticolinear", "orthogonal"] = "anticolinear",
        bimamba_type="none",
        if_divide_out=False
    ) -> "BiMambaLHeadModelWithPEFeatureExtractor":

        model = cls(
            config=config,
            whole_position_encoding=whole_position_encoding,
            shuffle=shuffle,
            total_seq_len=total_seq_len,
            embedding_init=embedding_init,
            embedding_init_mode=embedding_init_mode,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            pos_pred=pos_pred,
            bimamba_type=bimamba_type,
            if_divide_out=if_divide_out
        )
        pretrained_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k and "lm_head" not in k and "whole_position_encoding" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            print(f"The following keys were missing: {missing_keys}")
        if unexpected_keys:
            print(f"The following keys were unexpected: {unexpected_keys}")

        return model
