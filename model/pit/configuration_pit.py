from transformers import PretrainedConfig, CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PITConfig(PretrainedConfig):
    model_type = "pit"
    is_composition = True

    def __init__(
        self,
        protein_config=None,
        text_config=None,
        ignore_index=-100,
        protein_token_index=32000,
        projector_hidden_act="gelu",
        protein_feature_select_strategy="full",
        protein_feature_layer=-2,
        vocab_size=32000,
        use_moe=False,
        moe_config=None,
        sequence_only=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.protein_token_index = protein_token_index
        self.projector_hidden_act = projector_hidden_act
        self.protein_feature_select_strategy = protein_feature_select_strategy
        self.protein_feature_layer = protein_feature_layer
        self.vocab_size = vocab_size

        self.protein_config = protein_config

        if isinstance(self.protein_config, dict):
            protein_config["pretrained_model_name_or_path"] = protein_config[
                "_name_or_path"
            ]  # hack fix
            self.protein_config = AutoConfig.from_pretrained(**protein_config)
        elif protein_config is None:
            self.protein_config = CONFIG_MAPPING["esm"]()

        self.text_config = text_config

        if isinstance(self.text_config, dict):
            text_config["pretrained_model_name_or_path"] = text_config[
                "_name_or_path"
            ]  # hack fix
            self.text_config = AutoConfig.from_pretrained(**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        self.hidden_sizes = [
            self.protein_config.hidden_size,
            self.text_config.hidden_size,
        ]

        self.sequence_only = sequence_only
        self.use_moe = use_moe
        self.moe_config = (
            {
                "num_experts": 4,
                "ep_size": 4,
                "k": 1,
                "capacity_factor": 1.25,
                "eval_capacity_factor": 2.0,
                "use_residual": False,
                "drop_tokens": True,
                "use_rts": False,
                "use_tutel": True,
                "router_aux_loss_coef": 0.01,
            }
            if moe_config is None
            else moe_config
        )

        super().__init__(**kwargs)
