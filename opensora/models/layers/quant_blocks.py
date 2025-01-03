import torch
from timm.models.vision_transformer import Mlp
from torch import nn

from opensora.models.layers.blocks import Attention, MultiHeadCrossAttention
from opensora.utils.custom.operators import triton_flash_attn_bhsd
from opensora.utils.custom.operators.xformers import padded_xformers_attn
from opensora.utils.custom.triton_kernels.gelu_with_dynamic_quant import (
    gelu_with_fp8_dynamic_quant,
    gelu_with_int8_dynamic_quant,
)
from opensora.utils.custom.triton_kernels.layer_norm_with_dynamic_quant import (
    layer_norm_with_fp8_dynamic_quant,
    layer_norm_with_int8_dynamic_quant,
)
from opensora.utils.custom.triton_kernels.triton_scaled_mm import triton_scaled_mm
from opensora.utils.custom.triton_kernels.utils import cast_to_fp8_nearest


class QuantLinear(nn.Linear):
    def __init__(self, name: str = None, quant_mode: str = "int8", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.quant_mode = quant_mode

    @classmethod
    def from_original_module(
        cls, module: nn.Linear, name: str = None, quant_mode: str = "int8", smooth_scale: torch.Tensor = None
    ):
        new_module = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            name=name,
            quant_mode=quant_mode,
        )
        if smooth_scale is None:
            smooth_scale = torch.ones(module.weight.shape[1], dtype=torch.float32)
        smooth_scale = smooth_scale.to(module.weight.device).reshape(-1, 1)
        max_abs_val = torch.max(torch.abs(module.weight.T * smooth_scale), axis=0).values.to(torch.float32)

        finfo = torch.finfo(torch.float8_e4m3fnuz)
        iinfo = torch.iinfo(torch.int8)

        assert quant_mode in ["fp8", "int8"]

        # Forcely remove weight and bias from new_module to recreate them as tensor instead of parameter.
        del new_module.weight, new_module.bias

        if new_module.quant_mode == "fp8":
            new_module.weight_scale = max_abs_val / finfo.max
            new_module.weight_scale = new_module.weight_scale.contiguous()
            new_module.weight = cast_to_fp8_nearest(
                (module.weight.T * smooth_scale / new_module.weight_scale).clamp(min=finfo.min, max=finfo.max).T
            ).T
        elif new_module.quant_mode == "int8":
            new_module.weight_scale = max_abs_val / iinfo.max
            new_module.weight_scale = new_module.weight_scale.contiguous()
            new_module.weight = (
                torch.round(module.weight.T * smooth_scale / new_module.weight_scale)
                .clamp(min=iinfo.min, max=iinfo.max)
                .to(torch.int8)
            )

        new_module.weight_scale = new_module.weight_scale.to(torch.float32)
        new_module.bias = module.bias
        return new_module

    @torch.no_grad()
    def forward(self, input: torch.Tensor, input_scales: torch.Tensor = None) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]
        if self.quant_mode == "fp8":
            outputs = triton_scaled_mm(
                input=input_2d,
                weight=self.weight,
                scale_input=input_scales,
                scale_weight=self.weight_scale,
                bias=self.bias,
                out_dtype=torch.float16,
            )
        elif self.quant_mode == "int8":
            outputs = triton_scaled_mm(
                input=input_2d,
                weight=self.weight,
                scale_input=input_scales,
                scale_weight=self.weight_scale,
                bias=self.bias,
                out_dtype=torch.float16,
            )
        else:
            raise Exception(f"The forward pass of this quant mode `{self.quant_mode}` is not implemented yet.")

        output = outputs
        if isinstance(output, tuple):
            output, scales = outputs
            return output.view(*output_shape), scales.view(*output_shape[:-1])

        return output.view(*output_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}"


class GELUWithScale(nn.Module):
    def __init__(self, quant_mode: str = "int8") -> None:
        super().__init__()
        self.quant_mode = quant_mode
        if quant_mode == "int8":
            self.func = gelu_with_int8_dynamic_quant
        elif quant_mode == "fp8":
            self.func = gelu_with_fp8_dynamic_quant
        else:
            raise ValueError(f"Invalid quant_mode: {quant_mode}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)

    def extra_repr(self) -> str:
        return f"approximate=tanh, quant_mode={self.quant_mode}"


class CustomLayerNormWithScale(nn.LayerNorm):
    def __init__(self, name: str = "", quant_mode: str = "int8", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_mode = quant_mode
        self.name = name
        if quant_mode == "int8":
            self.func = layer_norm_with_int8_dynamic_quant
        elif quant_mode == "fp8":
            self.func = layer_norm_with_fp8_dynamic_quant
        else:
            raise ValueError(f"Invalid quant_mode: {quant_mode}")

    @classmethod
    def from_original_module(cls, module: nn.LayerNorm, name: str = "", quant_mode: str = "int8"):
        new_module = cls(
            name=name,
            quant_mode=quant_mode,
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
            bias=module.bias is not None,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
        return new_module

    def forward(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        scale_zero: torch.Tensor = None,
        shift_zero: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.func(
            input, self.normalized_shape, scale, shift, scale_zero, shift_zero, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return f"quant_mode={self.quant_mode}"


class QuantAttention(Attention):
    def __init__(self, name: str = "", quant_mode: str = "int8", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.quant_mode = quant_mode

    @classmethod
    def from_original_module(
        cls, module: nn.LayerNorm, name: str = None, quant_mode: str = "int8", qkv_scale: torch.Tensor = None
    ):
        dim = module.dim
        num_heads = module.num_heads
        qkv_bias = module.qkv.bias is not None
        qk_norm = not isinstance(module.q_norm, nn.Identity)
        qk_norm_legacy = module.qk_norm_legacy
        attn_drop = module.attn_drop.p
        proj_drop = module.proj_drop.p
        enable_flash_attn = module.enable_flash_attn
        rope = module.rotary_emb if module.rope else None
        new_module = cls(
            name=name,
            quant_mode=quant_mode,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash_attn=enable_flash_attn,
            rope=rope,
            qk_norm_legacy=qk_norm_legacy,
        )

        if getattr(module, "rotary_emb", None) is not None:
            new_module.rotary_emb = module.rotary_emb

        new_module.qkv = QuantLinear.from_original_module(
            module.qkv, name=name + ".qkv", quant_mode=quant_mode, smooth_scale=qkv_scale
        )
        new_module.old_qkv = module.qkv
        new_module.q_norm = module.q_norm
        new_module.k_norm = module.k_norm
        new_module.proj = module.proj

        return new_module

    def forward(self, x: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x, sx)
        # old_qkv = self.old_qkv((x*sx.reshape(x.shape[0],-1,1)).to(torch.float16))
        # qkv = old_qkv
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            # old torch-impl attn
            # dtype = q.dtype
            # q = q * self.scale
            # attn = q @ k.transpose(-2, -1)  # translate attn to float32
            # attn = attn.to(torch.float32)
            # if self.is_causal:
            #     causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
            #     causal_mask = torch.where(causal_mask.bool(), 0, float("-inf"))
            #     attn += causal_mask
            # attn = attn.softmax(dim=-1)
            # attn = attn.to(dtype)  # cast back attn to original dtype
            # attn = self.attn_drop(attn)
            # x = attn @ v
            # x = x.transpose(1, 2)  # transpose to 'bshd'

            # triton-bhsd
            x = triton_flash_attn_bhsd(q, k, v)
            x = x.transpose(1, 2)  # transpose to 'bshd'

            # triton-bshd
            # q = q.transpose(1, 2)
            # k = k.transpose(1, 2)
            # v = v.transpose(1, 2)
            # x = triton_flash_attn_bshd(q, k, v)

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(self, name: str = "", quant_mode: str = "int8", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.quant_mode = quant_mode

    @classmethod
    def from_original_module(
        cls,
        module: MultiHeadCrossAttention,
        name: str = None,
        quant_mode: str = "int8",
        q_linear_scale: torch.Tensor = None,
    ):
        new_module = cls(
            d_model=module.d_model,
            num_heads=module.num_heads,
            attn_drop=module.attn_drop.p,
            proj_drop=module.proj_drop.p,
            name=name,
            quant_mode=quant_mode,
        )

        new_module.q_linear = QuantLinear.from_original_module(
            module.q_linear, name=name + ".q_linear", quant_mode=quant_mode, smooth_scale=q_linear_scale
        )
        new_module.kv_linear = (
            module.kv_linear
        )  # QuantLinear.from_original_module(module.kv_linear, name=name + ".kv_linear", quant_mode=quant_mode)
        new_module.proj = module.proj

        return new_module

    def forward(self, x, k, v, sx, attn_bias=None):
        # torch.save(self.kv_linear.state_dict(), "/remote/vast0/share-mv/tien/project/Open-Sora/save/weights/kv_linear.{}.pth".format(self.name))

        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        q = self.q_linear(x, sx).view(1, -1, self.num_heads, self.head_dim)
        # kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        # k, v = kv.unbind(2)

        ### Calculate cross attn ###

        # xformers default impls
        # if enable_xformers:
        #     attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(1, 16, 216000, 600)
        #     x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        #     x = x.view(B, -1, C)
        # else:
        #     x = memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        #     x = x.reshape(B, -1, C)

        # pytorch default impls
        # attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(1, 16, 216000, 600)
        # q = q.transpose(1, 2) # transpose to 'bhsd' layout
        # k = k.transpose(1, 2) # transpose to 'bhsd' layout
        # v = v.transpose(1, 2) # transpose to 'bhsd' layout
        # x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        # x = x.transpose(1, 2) # transpose to 'bshd' layout
        # x = x.reshape(B, -1, C)

        # padded xformers default
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(1, 16, 216000, 600)
        x = padded_xformers_attn(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, -1, C)

        # normal tensor is not contiguous for view function
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class QuantMlp(Mlp):
    def __init__(self, name: str = "", quant_mode: str = "int8", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.quant_mode = quant_mode

    @classmethod
    def from_original_module(
        cls, module: nn.LayerNorm, name: str = None, quant_mode: str = "int8", fc1_scale: torch.Tensor = None
    ):
        in_features = module.fc1.in_features
        hidden_features = module.fc1.out_features
        out_features = module.fc2.out_features
        norm_layer = module.norm
        bias = module.fc1.bias is not None
        drop = module.drop1.p, module.drop2.p
        new_module = cls(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            bias=bias,
            drop=drop,
            name=name,
            quant_mode=quant_mode,
        )

        if not isinstance(module.act, nn.GELU):
            raise NotImplementedError("Activation layer in MLP should be GELU")
        else:
            if module.act.approximate != "tanh":
                raise NotImplementedError("Only tanh approximate is supported")
        new_module.act = GELUWithScale(quant_mode=quant_mode)

        new_module.fc1 = QuantLinear.from_original_module(
            module.fc1, name=name + ".fc1", quant_mode=quant_mode, smooth_scale=fc1_scale
        )
        new_module.fc2 = QuantLinear.from_original_module(module.fc2, name=name + ".fc2", quant_mode=quant_mode)

        if not isinstance(module.norm, nn.Identity):
            raise NotImplementedError("Norm layer in MLP should be Identity")
        new_module.norm = module.norm
        return new_module

    def forward(self, x, sx):
        x = self.fc1(x, sx)
        x, sx = self.act(x)
        x = self.fc2(x, sx)
        return x
