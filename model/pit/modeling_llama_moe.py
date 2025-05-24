#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from deepspeed.moe.layer import MoE
from safetensors import safe_open
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache, StaticCache, Cache
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput

from model.pit.modeling_llama import LlamaForCausalLM, LlamaMLP


class MoELlamaConfig(LlamaConfig):
    model_type = "moe_llama"

    def __init__(
        self,
        num_experts=2,
        ep_size=2,
        k=1,
        capacity_factor=1.25,
        eval_capacity_factor=2.0,
        use_residual=True,
        drop_tokens=True,
        use_rts=True,
        use_tutel=True,
        router_aux_loss_coef=0.01,
        **kwargs,
    ):
        self.moe_config = {
            "num_experts": num_experts,
            "ep_size": ep_size,
            "k": k,
            "capacity_factor": capacity_factor,
            "eval_capacity_factor": eval_capacity_factor,
            "use_residual": use_residual,
            "drop_tokens": drop_tokens,
            "use_rts": use_rts,
            "use_tutel": use_tutel,
        }
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss_list: Optional[Tuple[torch.FloatTensor]] = None


def MoELlamaDecoderLayer_forward(self):
    def forward(
        # self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        aux_losses = []
        if len(hidden_states) == 3:
            aux_losses.append(hidden_states[1])
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (aux_losses,)

        return outputs

    return forward


def MoELlamaModel_forward(self):
    def forward(
        # self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_aux_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "cache_position is a required argument when using StaticCache."
                )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_aux_loss = [] if output_aux_loss else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_aux_loss:
                all_aux_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_loss_list=all_aux_loss,
        )

    return forward


class MoELlamaModel(LlamaModel):
    config_class = MoELlamaConfig

    def __init__(self, config):
        super(MoELlamaModel, self).__init__(config)


class MoELlamaForCausalLM(LlamaForCausalLM):
    config_class = MoELlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoELlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss, aux_losses = None, []
        if len(outputs[-1]) > 0:
            aux_loss_list = outputs[-1]
            for aux_loss in aux_loss_list:
                if aux_loss is not None:
                    aux_losses.append(aux_loss)
            aux_loss = self.router_aux_loss_coef * sum(aux_losses)
            if labels is not None:
                print(loss, sum(aux_losses), loss + aux_loss)
                loss += aux_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (aux_loss,) + output if aux_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss_list=outputs.aux_loss_list,
        )

    def initialize_moe_modules(self, moe_config, router_aux_loss_coef):
        self.num_experts = self.config.num_experts = moe_config["num_experts"]
        self.router_aux_loss_coef = self.config.router_aux_loss_coef = (
            router_aux_loss_coef
        )
        self.moe_config = self.config.moe_config = moe_config
        num_layers = self.config.num_hidden_layers

        for i in range(num_layers):
            pretrained_state_dict = self.model.layers[i].mlp.state_dict()
            self.model.layers[i].mlp = MoE(
                self.config.hidden_size,
                expert=self.model.layers[i].mlp,
                **self.config.moe_config,
            )
            for e in self.model.layers[
                i
            ].mlp.deepspeed_moe.experts.deepspeed_experts:  # check weight
                loaded_state_dict = e.state_dict()
                assert all(
                    [
                        torch.allclose(pretrained_state_dict[k], v)
                        for k, v in loaded_state_dict.items()
                    ]
                )
                assert all(
                    [
                        torch.allclose(loaded_state_dict[k], v)
                        for k, v in pretrained_state_dict.items()
                    ]
                )
        for m in self.model.layers:
            m.forward = MoELlamaDecoderLayer_forward(m)
        self.model.forward = MoELlamaModel_forward(self.model)


class EvalSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.moe_config["num_experts"]
        self.top_k = config.moe_config["k"]

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [LlamaMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class EvalMoELlamaForCausalLM(LlamaForCausalLM):
    config_class = MoELlamaConfig

    def __init__(self, config):
        super(EvalMoELlamaForCausalLM, self).__init__(config)
        num_layers = self.config.num_hidden_layers
        self.num_experts = self.config.num_experts = config.moe_config["num_experts"]
        for i in range(num_layers):
            self.model.layers[i].mlp = EvalSparseMoeBlock(self.config)

    def load_experts_from_deepspeed_ckpts(self, ckpt_path):
        import os
        import warnings

        entries = os.listdir(ckpt_path)
        deepspeed_ckpt_path = [
            entry
            for entry in entries
            if entry.startswith("global_")
            and os.path.isdir(os.path.join(ckpt_path, entry))
        ]

        if len(deepspeed_ckpt_path) == 0:
            warnings.warn(
                "No DeepSpeed checkpoint found in the provided path (no 'global_' directory)"
            )
            warnings.warn(
                "Make sure the weights of experts & gates are loaded from the safetensors file"
            )
            return

        if len(deepspeed_ckpt_path) > 1:
            raise ValueError(
                "Multiple DeepSpeed checkpoints found in the provided path"
            )

        deepspeed_ckpt_path = os.path.join(ckpt_path, deepspeed_ckpt_path[0])
        ckpt_format = "layer_{l}_expert_{e}_mp_rank_00_model_states.pt"

        print("Loading experts and gates from DeepSpeed checkpoint...")
        for l in range(self.config.num_hidden_layers):
            print(f"Loading gate of layer {l}")
            with safe_open(
                os.path.join(ckpt_path, "model.safetensors"),
                framework="pt",
                device="cpu",
            ) as f:
                self.model.layers[l].mlp.gate.weight.data = f.get_tensor(
                    f"text_model.model.layers.{l}.mlp.deepspeed_moe.gate.wg.weight"
                ).to(
                    device=self.model.layers[l].mlp.gate.weight.data.device,
                    dtype=self.model.layers[l].mlp.gate.weight.data.dtype,
                )
            for e in range(self.num_experts):
                print(f"Loading expert {e} of layer {l}")
                ckpt = torch.load(
                    os.path.join(deepspeed_ckpt_path, ckpt_format.format(l=l, e=e)),
                    map_location="cpu",
                )
                self.model.layers[l].mlp.experts[e].gate_proj.weight.data = ckpt[
                    f"text_model.model.layers.{l}.mlp.deepspeed_moe.experts.deepspeed_experts.{e}.gate_proj.weight"
                ].to(
                    device=self.model.layers[l]
                    .mlp.experts[e]
                    .gate_proj.weight.data.device,
                    dtype=self.model.layers[l]
                    .mlp.experts[e]
                    .gate_proj.weight.data.dtype,
                )
                self.model.layers[l].mlp.experts[e].up_proj.weight.data = ckpt[
                    f"text_model.model.layers.{l}.mlp.deepspeed_moe.experts.deepspeed_experts.{e}.up_proj.weight"
                ].to(
                    device=self.model.layers[l]
                    .mlp.experts[e]
                    .up_proj.weight.data.device,
                    dtype=self.model.layers[l].mlp.experts[e].up_proj.weight.data.dtype,
                )
                self.model.layers[l].mlp.experts[e].down_proj.weight.data = ckpt[
                    f"text_model.model.layers.{l}.mlp.deepspeed_moe.experts.deepspeed_experts.{e}.down_proj.weight"
                ].to(
                    device=self.model.layers[l]
                    .mlp.experts[e]
                    .down_proj.weight.data.device,
                    dtype=self.model.layers[l]
                    .mlp.experts[e]
                    .down_proj.weight.data.dtype,
                )
