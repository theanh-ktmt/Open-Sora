import os

import torch
import torch.nn as nn
from einops import rearrange

from opensora.models.layers.quant_blocks import (
    CustomLayerNormWithScale,
    QuantAttention,
    QuantMlp,
    QuantMultiHeadCrossAttention,
)
from opensora.models.stdit.stdit3 import STDiT3Block
from opensora.utils.custom.triton_kernels.add_and_scale import add_and_scale
from opensora.utils.misc import create_logger

logger = create_logger()

activations_dict = None
activations_fp = "/remote/vast0/share-mv/tien/project/Open-Sora/save/quantization/activations_dict.pth"
if os.path.exists(activations_fp):
    activations_dict = torch.load(activations_fp)


def custom_t_mask_select(x_mask, x, masked_x, T, S):
    if x_mask.sum() == x_mask.shape[0] * x_mask.shape[1]:
        return x
    x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
    masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
    x = torch.where(x_mask[:, :, None, None], x, masked_x)
    x = rearrange(x, "B T S C -> B (T S) C")
    return x


class QuantSTDiT3Block(STDiT3Block):
    def __init__(self, name: str = None, quant_mode: str = "int8", use_smoothquant: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.quant_mode = quant_mode
        self.step = 0

        if use_smoothquant:
            assert activations_dict is not None, "activations_dict is None"
        self.use_smoothquant = use_smoothquant

    @classmethod
    def from_original_module(
        cls,
        module: nn.Linear,
        name: str = None,
        quant_mode: str = "int8",
        use_smoothquant: bool = False,
        alpha: float = 0.5,
    ):
        block_index = module.block_index
        temporal = module.temporal
        hidden_size = module.hidden_size
        num_heads = module.attn.num_heads
        enable_flash_attn = module.enable_flash_attn
        enable_sequence_parallelism = module.enable_sequence_parallelism

        new_module = cls(
            name=name,
            quant_mode=quant_mode,
            use_smoothquant=use_smoothquant,
            hidden_size=hidden_size,
            num_heads=num_heads,
            block_index=block_index,
            temporal=temporal,
            enable_flash_attn=enable_flash_attn,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )

        norm1_scale, norm2_scale, new_module.residual_smooth_scale = None, None, None
        if use_smoothquant:
            attn_qkv_act = activations_dict[f"{name}.attn.qkv"]
            attn_qkv_max_weight = module.attn.qkv.weight.max(dim=0).values
            norm1_scale = (attn_qkv_max_weight / attn_qkv_act) ** alpha

            mlp_fc1_act = activations_dict[f"{name}.mlp.fc1"]
            mlp_fc1_max_weight = module.mlp.fc1.weight.max(dim=0).values
            norm2_scale = (mlp_fc1_max_weight / mlp_fc1_act) ** alpha

            cross_attn_q_linear_act = activations_dict[f"{name}.cross_attn.q_linear"]
            cross_attn_q_linear_max_weight = module.cross_attn.q_linear.weight.max(dim=0).values
            new_module.residual_smooth_scale = (cross_attn_q_linear_max_weight / cross_attn_q_linear_act) ** alpha

        new_module.norm1 = CustomLayerNormWithScale.from_original_module(
            module.norm1, name=f"{name}.norm1", quant_mode=quant_mode
        )
        new_module.norm2 = CustomLayerNormWithScale.from_original_module(
            module.norm2, name=f"{name}.norm2", quant_mode=quant_mode
        )

        if use_smoothquant:
            del new_module.norm1.weight, new_module.norm2.weight
            new_module.norm1.weight = norm1_scale
            new_module.norm2.weight = norm2_scale

        new_module.attn = QuantAttention.from_original_module(
            module=module.attn, name=f"{name}.attn", quant_mode=quant_mode, qkv_scale=1 / norm1_scale
        )
        if module.enable_sequence_parallelism:
            raise NotImplementedError("Sequence parallelism is not supported in quantized mode")
        new_module.cross_attn = QuantMultiHeadCrossAttention.from_original_module(
            module=module.cross_attn,
            name=f"{name}.cross_attn",
            quant_mode=quant_mode,
            q_linear_scale=1 / new_module.residual_smooth_scale,
        )
        new_module.mlp = QuantMlp.from_original_module(
            module=module.mlp, name=f"{name}.mlp", quant_mode=quant_mode, fc1_scale=1 / norm2_scale
        )

        new_module.drop_path = module.drop_path
        if not isinstance(new_module.drop_path, nn.Identity):
            raise NotImplementedError("Only Identity drop path is supported in quantized mode")
        new_module.scale_shift_table = module.scale_shift_table

        if quant_mode == "fp8":
            new_module.t_mask_select = custom_t_mask_select

        return new_module

    def forward(
        self,
        x,
        t,
        mha_key,
        mha_value,
        mha_bias,
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        self.step += 1

        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        shift_msa_zero, scale_msa_zero, shift_mlp_zero, scale_mlp_zero = None, None, None, None
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        # modulate (attention)
        x_m, x_m_zero, scale = self.norm1(x, scale_msa, shift_msa, scale_msa_zero, shift_msa_zero)
        if x_mask is not None:
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m, scale)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m, scale)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x, sx, org_x = add_and_scale(x, x_m_s, self.residual_smooth_scale)  # self.drop_path(x_m_s)

        # cross attention
        x = org_x + self.cross_attn(x, mha_key, mha_value, sx, mha_bias)

        # modulate (MLP)
        x_m, x_m_zero, scale = self.norm2(x, scale_mlp, shift_mlp, scale_mlp_zero, shift_mlp_zero)
        if x_mask is not None:
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m, scale)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x += x_m_s

        return x
