import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput

from model.pit.configuration_pit import PITConfig
from model.pit.modeling_llama import LlamaForCausalLM
from model.pit.modeling_llama_moe import MoELlamaForCausalLM, EvalMoELlamaForCausalLM


@dataclass
class PITCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    protein_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PITMultiModalProjector(nn.Module):
    def __init__(self, config: PITConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.protein_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(self, protein_features):
        hidden_states = self.linear_1(protein_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PITPreTrainedModel(PreTrainedModel):
    config_class = PITConfig
    base_model_prefix = "pit"
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    @property
    def _supports_sdpa(self):
        """
        Retrieve text_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.text_model._supports_sdpa


class PITForConditionGeneration(PITPreTrainedModel):
    def __init__(self, config: PITConfig):
        super().__init__(config)
        self.sequence_only = config.sequence_only
        if self.sequence_only:
            from transformers import EsmModel
        else:
            from model.esm_structure.modeling_esm_structure import EsmModel

        if config.use_moe:
            self.text_model = MoELlamaForCausalLM(config.text_config)
            self.protein_model = EsmModel(config.protein_config)
        else:
            self.protein_model = EsmModel.from_pretrained(
                config.protein_config._name_or_path
            )
            self.text_model = LlamaForCausalLM.from_pretrained(
                config.text_config._name_or_path
            )
        self.protein_model.gradient_checkpointing_enable(
            {"use_reentrant": not config.use_moe}
        )
        self.text_model.gradient_checkpointing_enable(
            {"use_reentrant": not config.use_moe}
        )
        self.multi_modal_projector = PITMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else self.config.eos_token_id
        )
        self.resize_token_embeddings(self.vocab_size, pad_to_multiple_of=8)
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.text_model.set_decoder(decoder)

    def get_decoder(self):
        return self.text_model.get_decoder()

    def tie_weights(self):
        return self.text_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.text_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_protein_features(
        self, protein_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        num_proteins, protein_sequence_length, embed_dim = protein_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id)
        )
        # 1. Create a mask to know where special protein tokens are
        special_protein_token_mask = input_ids == self.config.protein_token_index
        num_special_protein_tokens = torch.sum(special_protein_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_protein_tokens.max() * (protein_sequence_length - 1)
        ) + sequence_length
        batch_indices, non_protein_indices = torch.where(
            input_ids != self.config.protein_token_index
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged protein-text sequence.
        # `special_protein_token_mask` identifies protein tokens. Each protein token will be replaced by `nb_text_tokens_per_proteins - 1` text tokens.
        # `torch.cumsum` computes how each protein token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum(
                (special_protein_token_mask * (protein_sequence_length - 1) + 1), -1
            )
            - 1
        )
        nb_protein_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_protein_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_protein_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                self.config.ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Protein model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_protein_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_protein_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<protein>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the protein features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_protein_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_protein_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_protein_indices
            ]

        # 5. Fill the embeddings corresponding to the proteins. Anything that is still zeros needs filling
        protein_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        protein_to_overwrite &= protein_to_overwrite.cumsum(-1) - 1 >= nb_protein_pad[
            :, None
        ].to(target_device)

        if protein_to_overwrite.sum() != protein_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of protein tokens is {torch.sum(special_protein_token_mask)} while"
                f" the number of protein given to the model is {num_proteins}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[protein_to_overwrite] = (
            protein_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= protein_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        node_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        protein_feature_layer: Optional[int] = None,
        protein_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PITCausalLMOutputWithPast]:
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
        protein_feature_layer = (
            protein_feature_layer
            if protein_feature_layer is not None
            else self.config.protein_feature_layer
        )
        protein_feature_select_strategy = (
            protein_feature_select_strategy
            if protein_feature_select_strategy is not None
            else self.config.protein_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and proteins
            if (
                protein_input_ids is not None
                and protein_attention_mask is not None
                and input_ids.shape[1] != 1
            ):
                if not self.sequence_only:
                    protein_outputs = self.protein_model(
                        input_ids=protein_input_ids,
                        attention_mask=protein_attention_mask,
                        node_position=node_position,
                        output_hidden_states=True,
                    )
                else:
                    protein_outputs = self.protein_model(
                        input_ids=protein_input_ids,
                        attention_mask=protein_attention_mask,
                        output_hidden_states=True,
                    )
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_protein_feature = protein_outputs.hidden_states[
                    protein_feature_layer
                ]

                if protein_feature_select_strategy == "default":
                    selected_protein_feature = selected_protein_feature[:, 1:]
                elif protein_feature_select_strategy == "full":
                    selected_protein_feature = selected_protein_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.protein_feature_select_strategy}"
                    )

                protein_features = self.multi_modal_projector(selected_protein_feature)
                inputs_embeds, attention_mask, labels, position_ids = (
                    self._merge_input_ids_with_protein_features(
                        protein_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                    )
                )
                if labels is None:
                    labels = torch.full_like(
                        attention_mask, self.config.ignore_index
                    ).to(torch.long)
            # In case input_ids.shape[1] == 1 & protein_input_ids==None & past_key_values != None, we are in the case of
            # generation with cache
            elif (
                past_key_values is not None
                and protein_input_ids is not None
                and input_ids.shape[1] == 1
            ):
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PITCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        protein_input_ids=None,
        protein_attention_mask=None,
        node_position=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.protein_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "protein_input_ids": protein_input_ids,
                "protein_attention_mask": protein_attention_mask,
                "node_position": node_position,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.text_model._reorder_cache(*args, **kwargs)


class EvalPITForConditionGeneration(PITForConditionGeneration):
    _no_split_modules = ["text_model.lm_head.weight"]

    def __init__(self, config: PITConfig):
        PreTrainedModel.__init__(self, config)
        config.text_config.router_aux_loss_coef = config.moe_config.pop(
            "router_aux_loss_coef", 0.01
        )
        config.text_config.moe_config = config.moe_config
        self.num_experts = config.moe_config["num_experts"]
        self.num_hidden_layers = config.text_config.num_hidden_layers
        self.sequence_only = config.sequence_only
        if self.sequence_only:
            from transformers import EsmModel
        else:
            from model.esm_structure.modeling_esm_structure import EsmModel
        self.protein_model = EsmModel(config.protein_config)
        if config.use_moe:
            self.text_model = EvalMoELlamaForCausalLM(config.text_config)
        else:
            self.text_model = LlamaForCausalLM(config.text_config)
        # self.text_model.gradient_checkpointing_enable({'use_reentrant': False})
        self.multi_modal_projector = PITMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else self.config.eos_token_id
        )
        # self.resize_token_embeddings(self.vocab_size, pad_to_multiple_of=8)
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        if hasattr(model.text_model, "load_experts_from_deepspeed_ckpts"):
            model.text_model.load_experts_from_deepspeed_ckpts(
                ckpt_path=pretrained_model_name_or_path
            )
        return model
