import numpy as np
import torch
import torch.nn as nn
from torch.distributed import get_rank, get_world_size
from torch.nn.functional import cross_entropy, mse_loss
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel

from model.esm_structure.modeling_esm_structure import EsmModel
from model.esm_structure.structure_bias import PositionHead
from model.task_model.layers import CLIPLoss


class ProteinTextCLIPConfig(PretrainedConfig):
    model_type = "protein_text_clip"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 projection_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)


class ProteinTextCLIPForPretrain(PreTrainedModel):
    config_class = ProteinTextCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        self.protein_model = EsmModel.from_pretrained(
            protein_model_config._name_or_path)  # use this line if you want to train from scratch
        # self.protein_model = EsmModel(protein_model_config)
        self.protein_model.gradient_checkpointing_enable()
        self.text_model = AutoModel.from_pretrained(
            text_model_config._name_or_path)  # use this line if you want to train from scratch
        # self.text_model = BertModel(text_model_config)
        # self.text_model.gradient_checkpointing_enable({'use_reentrant': False})

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.GELU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.GELU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

        self.pos_head = PositionHead(
            embed_dim=protein_model_config.hidden_size,
            num_heads=protein_model_config.num_attention_heads
        )

        self.mlm_head = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.projection_dim),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

    def forward(self,
                protein_input_ids,
                protein_attention_mask,
                node_position,
                text_input_ids,
                text_attention_mask,
                protein_masked_input_ids,
                protein_masked_labels
                ):
        protein_embeds = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask, node_position=node_position
        ).last_hidden_state.mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state.mean(dim=1)
        text_embeds = self.text_projection(text_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        cl_loss = CLIPLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=get_rank() if torch.distributed.is_initialized() else 0,
            world_size=get_world_size() if torch.distributed.is_initialized() else 1
        )(protein_embeds, text_embeds, self.logit_scale.exp())

        noise_mask = protein_attention_mask.clone()
        counts = (1 - noise_mask).sum(dim=1)
        indices = noise_mask.size(1) - counts
        row_indices = torch.arange(noise_mask.size(0))
        noise_mask[row_indices, indices - 1] = 0  # <eos> token
        noise_mask[:, 0] = 0  # <cls> token
        noise_mask[protein_masked_labels != -100] = 0  # <mask> token
        noise = torch.randn_like(node_position) * noise_mask.unsqueeze(-1).float()
        noised_node_position = node_position + noise * 0.2  # Fixme: hard-coded noise scale

        protein_perturbed_output = self.protein_model(
            input_ids=protein_masked_input_ids,
            attention_mask=protein_attention_mask,
            node_position=noised_node_position,
        )

        noise_pred = self.pos_head(
            query=protein_perturbed_output.last_hidden_state,
            attention_bias=protein_perturbed_output.attention_bias,
            node_position=noised_node_position
        )

        noise_pred.masked_fill_((1 - noise_mask).unsqueeze(-1).to(torch.bool), 0.0)
        pos_loss = mse_loss(noise_pred, noise, reduction="sum")
        pos_loss = pos_loss / noise_mask.sum() / 3

        mlm_logits = self.mlm_head(protein_perturbed_output.last_hidden_state)
        mlm_loss = cross_entropy(mlm_logits.view(-1, mlm_logits.shape[-1]), protein_masked_labels.view(-1))

        return {
            "loss": cl_loss + pos_loss + mlm_loss,
            "cl_loss": cl_loss,
            "pos_loss": pos_loss,
            "mlm_loss": mlm_loss
        }
