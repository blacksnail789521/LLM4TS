import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoConfig
from peft import get_peft_config, get_peft_model, TaskType, LoraConfig, AdaLoraConfig
from einops import rearrange
from collections import OrderedDict
from layers.Embed import PatchEmbedding_temp
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2308.08469.pdf
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.C = args.enc_in if args.return_single_feature == False else 1

        # RevIN
        self.revin = RevIN(self.C, affine=True)

        # Input layer
        self._set_input_layer()

        # Output layer
        self._set_output_layer()

        # Model
        self._set_model()

    def _set_input_layer(self):
        if self.args.LLM == "gpt2":
            pos_embed_type = "none"
        else:
            # elif args.LLM == "llama":
            pos_embed_type = "learned"
        token_embed_type = (
            "linear"
            if getattr(self.args, "token_embed_type", None) is None
            else self.args.token_embed_type
        )
        temporal_embed_type = (
            "learned"
            if getattr(self.args, "temporal_embed_type", None) is None
            else self.args.temporal_embed_type
        )
        token_embed_kernel_size = (
            3
            if getattr(self.args, "token_embed_kernel_size", None) is None
            else self.args.token_embed_kernel_size
        )
        self.input_layer = PatchEmbedding_temp(
            self.args.C_t,
            self.args.d_model,
            self.args.patch_len,
            self.args.stride,
            self.args.dropout,
            pos_embed_type=pos_embed_type,
            token_embed_type=token_embed_type,
            kernel_size=token_embed_kernel_size,
            temporal_embed_type=temporal_embed_type,
            freq=self.args.freq,
        )  # (B * C, T_p, D)

    def _set_output_layer(self):
        if self.args.task_name == "supervised_finetuning":
            self.output_layer = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear",
                            nn.Linear(
                                self.args.d_model, self.args.patch_len, bias=False
                            ),
                        ),
                        ("dropout", nn.Dropout(self.args.dropout)),
                        # (B * C, T_p, P)
                    ]
                )
            )
        else:
            T_p = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 2)
            T_out = self.args.pred_len
            self.output_layer = nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", nn.Flatten(start_dim=1)),
                        # (B * C, T_p * D)
                        (
                            "linear",
                            nn.Linear(self.args.d_model * T_p, T_out, bias=False),
                        ),
                        # (B * C, T_out)
                        ("dropout", nn.Dropout(self.args.dropout)),
                        (
                            "rearrange",
                            RearrangeModule("(B C) T_out -> B T_out C", C=self.C),
                        ),
                        # (B, T_out, C)
                    ]
                )
            )

    def _set_model(self):
        # Load LLM
        if self.args.no_pretrain:
            config = AutoConfig.from_pretrained(self.args.LLM_path / "config.json")
            self.model = AutoModel.from_config(config)
            assert (
                self.args.no_freeze
            ), "If no_pretrain is True, no_freeze must be True."
        else:
            self.model = AutoModel.from_pretrained(self.args.LLM_path)

        # Only choose the first K layers
        self.model.h = self.model.h[: self.args.first_k_layers]

        # Apply PEFT
        peft_method = self.args.peft_method
        if peft_method == "none":
            print("No PEFT applied.")

            if not self.args.no_freeze:  # we should freeze
                # (self.model) Unfreeze the parameters of wpe, and freeze the others
                for name, param in self.model.named_parameters():
                    if any(term in name for term in ["wpe"]):
                        # if any(term in name for term in ["ln", "wpe"]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        else:
            print(f"Apply PEFT: {peft_method}")

            # Set peft_params
            if peft_method == "adalora" and self.args.peft_params_lora_dropout == 0:
                # adalora with dropout=0 is problematic
                self.args.peft_params_lora_dropout = 0.01
            peft_params = {
                "r": self.args.peft_params_r,
                "lora_alpha": self.args.peft_params_lora_alpha,
                "lora_dropout": self.args.peft_params_lora_dropout,
            }
            peft_params["task_type"] = TaskType.FEATURE_EXTRACTION
            peft_params["target_modules"] = ["c_attn"]
            peft_params["fan_in_fan_out"] = True

            # Use peft_params to get peft_config
            if peft_method == "lora":
                peft_config = LoraConfig(**peft_params)
            elif peft_method == "adalora":
                peft_config = AdaLoraConfig(**peft_params)
            else:
                raise NotImplementedError

            # Apply PEFT to model (all weights in self.model are frozen)
            self.model = get_peft_model(self.model, peft_config)

            if not self.args.no_freeze:  # we should freeze
                # (self.model) Only unfreeze the parameters of ln and wpe
                for name, param in self.model.named_parameters():
                    if any(term in name for term in ["ln", "wpe"]):
                        param.requires_grad = True
            else:
                # Unfreeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = True

        # Linear probing
        if "lp" in self.args.ft_mode:  # "lp" or "lp_ft"
            print("Apply linear probing.")
            # Freeze revin, input_layer and model
            for param in self.revin.parameters():
                param.requires_grad = False
            for param in self.input_layer.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = False

    def linear_probe_to_fine_tuning(self):
        print("Linear probing to fine-tuning.")
        # Unfreeze revin, input_layer and peft
        for param in self.revin.parameters():
            param.requires_grad = True
        for param in self.input_layer.parameters():
            param.requires_grad = True
        for name, param in self.model.named_parameters():
            if any(term in name for term in ["lora", "ia3"]):
                param.requires_grad = True

    def supervised_finetuning(self, x, x_mark):  # (B, T_in, C)
        # x: normalized
        # y: normalized + patched

        x = self.input_layer(x, x_mark)  # (B * C, T_p, D)
        y = self.model(inputs_embeds=x).last_hidden_state  # (B * C, T_p, D)
        y = self.output_layer(y)  # (B * C, T_p, P)

        return y

    def forecast(self, x, x_mark):  # (B, T_in, C)
        x = self.revin(x, "norm")  # (B, T_in, C)
        x = self.input_layer(x, x_mark)  # (B * C, T_p, D)
        y = self.model(inputs_embeds=x).last_hidden_state  # (B * C, T_p, D)
        y = self.output_layer(y)  # (B, T_out, C)
        y = self.revin(y, "denorm")
        return y

    def imputation(self, x, x_mark, mask):  # (B, T_in, C)
        x = self.revin(x, "norm", mask)  # (B, T_in, C) with mask
        x = self.input_layer(x, x_mark)  # (B * C, T_p, D)
        y = self.model(inputs_embeds=x).last_hidden_state  # (B * C, T_p, D)
        y = self.output_layer(y)  # (B, T_in, C)
        y = self.revin(y, "denorm")

        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.args.task_name == "supervised_finetuning":
            dec_out = self.supervised_finetuning(x_enc, x_mark_enc)
            return dec_out  # (B, T_out, C), T_out = T_in
        elif self.args.task_name == "long_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out  # (B, T_out, C)
        else:
            raise NotImplementedError
        return None
