import os

from datasets import load_dataset
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AddedToken, Trainer

from model.pit.configuration_pit import PITConfig
from model.pit.modeling_llama import LlamaForCausalLM
from model.pit.modeling_pit import PITForConditionGeneration
from model.task_model.pretrain_model_esm import ProteinTextCLIPConfig, ProteinTextCLIPForPretrain
from trainer import PITPretrainTrainer, CLIPPretrainTrainer
from utils import DataCollatorForPIT, DataCollatorForPureLLM, DataCollatorForProteinTextCLIPPretrain


class PretrainTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.task_model = self.build_task_model()
        self.dataset = self.build_dataset()
        self.train_args = self.build_train_args()
        self.trainer = self.build_trainer()

    def build_task_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        raise NotImplementedError()

    def build_train_args(self):
        raise NotImplementedError()

    def build_trainer(self):
        raise NotImplementedError()

    def run(self):
        self.trainer.train()


class ProteinTextCLIPPretrainTask(PretrainTask):
    def __init__(self, run_config):
        self.protein_model_config = AutoConfig.from_pretrained(run_config.protein_model_name)
        self.text_model_config = AutoConfig.from_pretrained(run_config.text_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(run_config.protein_model_name, use_fast=False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(run_config.text_model_name, use_fast=False)
        self.pdb_h5_path = os.path.join(run_config.data_path, 'pdb.h5')
        super().__init__(run_config)

    def build_task_model(self):
        task_model_config = ProteinTextCLIPConfig(
            protein_model_config=self.protein_model_config,
            text_model_config=self.text_model_config,
            projection_dim=self.run_config.projection_dim,
        )
        task_model = ProteinTextCLIPForPretrain(task_model_config)
        return task_model

    def build_dataset(self):
        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.jsonl',
        })
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            do_eval=False,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            fp16=self.run_config.fp16,
            push_to_hub=False,
            learning_rate=self.run_config.lr,
            report_to=["wandb"],
            warmup_ratio=self.run_config.warmup_ratio,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            deepspeed=self.run_config.deepspeed,
        )

    def build_trainer(self):
        return CLIPPretrainTrainer(
            model=self.task_model,
            args=self.train_args,
            data_collator=DataCollatorForProteinTextCLIPPretrain(self.protein_tokenizer,
                                                                 self.text_tokenizer,
                                                                 self.pdb_h5_path,
                                                                 sequence_only=self.run_config.sequence_only,
                                                                 mlm_probability=getattr(self.run_config,
                                                                                         "mlm_probability", 0.0), ),
            train_dataset=self.dataset["train"],
            protein_model_fixed=self.run_config.protein_model_fixed,
            text_model_fixed=self.run_config.text_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
        )


class PITStage1PretrainTask(PretrainTask):
    def __init__(self, run_config):
        self.protein_model_config = AutoConfig.from_pretrained(run_config.protein_model_name)
        self.text_model_config = AutoConfig.from_pretrained(run_config.text_model_name)

        # Load the tokenizers
        self.protein_tokenizer = AutoTokenizer.from_pretrained(run_config.protein_model_name, use_fast=False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(run_config.text_model_name, use_fast=False)
        # Add the protein token to the tokenizer
        self.text_tokenizer.add_tokens(AddedToken(run_config.protein_token, special=True, normalized=False),
                                       special_tokens=True)
        self.protein_token_index = self.text_tokenizer.convert_tokens_to_ids(run_config.protein_token)
        # Set the padding side to right
        self.text_tokenizer.padding_side = "right"
        # Add the special tokens for the conversation
        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bop>", "<eop>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]})

        # Save the tokenizers for evaluation
        self.protein_tokenizer.save_pretrained(f'{run_config.output_path}/protein_tokenizer')
        self.text_tokenizer.save_pretrained(f'{run_config.output_path}/text_tokenizer')

        self.pdb_h5_path = os.path.join(run_config.data_path, 'pdb.h5')

        super().__init__(run_config)

    def build_task_model(self):
        config = PITConfig(
            protein_config=self.protein_model_config,
            text_config=self.text_model_config,
            pad_token_id=self.text_tokenizer.pad_token_id,
            eos_token_id=self.text_tokenizer.eos_token_id,
            protein_token_index=self.protein_token_index,
            vocab_size=len(self.text_tokenizer),  # rather than self.text_tokenizer.vocab_size
            use_moe=self.run_config.text_model_moe,
            sequence_only=self.run_config.sequence_only,
        )
        config.save_pretrained(self.run_config.output_path)
        task_model = PITForConditionGeneration(config)
        return task_model

    def build_dataset(self):
        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.jsonl',
        })
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            do_eval=False,
            save_strategy="epoch",
            save_total_limit=5,
            logging_strategy="steps",
            logging_steps=10,
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            fp16=self.run_config.fp16,
            push_to_hub=False,
            remove_unused_columns=False,
            learning_rate=self.run_config.lr,
            report_to=["wandb"],
            warmup_ratio=self.run_config.warmup_ratio,
            deepspeed=self.run_config.deepspeed,
            dataloader_num_workers=4,
        )

    def build_trainer(self):
        return PITPretrainTrainer(
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            data_collator=DataCollatorForPIT(self.protein_tokenizer, self.text_tokenizer, self.run_config.max_length,
                                             self.pdb_h5_path, self.run_config.protein_token,
                                             sequence_only=self.run_config.sequence_only,
                                             drop_structure_rate=self.run_config.drop_structure_rate),
            protein_model_fixed=self.run_config.protein_model_fixed,
            text_model_fixed=self.run_config.text_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
            use_moe=self.run_config.text_model_moe,
        )


class PITStage2PretrainTask(PITStage1PretrainTask):
    def build_task_model(self):
        config = PITConfig.from_pretrained(self.run_config.pit_model_name, use_moe=self.run_config.text_model_moe)
        task_model = PITForConditionGeneration.from_pretrained(self.run_config.pit_model_name, config=config)
        if self.run_config.text_model_moe:
            router_aux_loss_coef = task_model.config.moe_config.pop("router_aux_loss_coef", 0.01)
            task_model.text_model.initialize_moe_modules(task_model.config.moe_config, router_aux_loss_coef)
        return task_model


class PureLLMPretrainTask(PITStage1PretrainTask):
    def __init__(self, run_config):
        self.text_model_config = AutoConfig.from_pretrained(run_config.text_model_name)

        # Load the tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(run_config.text_model_name, use_fast=False)
        # Add the protein token to the tokenizer
        self.text_tokenizer.add_tokens(AddedToken(run_config.protein_token, special=True, normalized=False),
                                       special_tokens=True)
        self.protein_token_index = self.text_tokenizer.convert_tokens_to_ids(run_config.protein_token)
        # Set the padding side to right
        self.text_tokenizer.padding_side = "right"
        # Add the special tokens for the conversation
        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bop>", "<eop>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]})

        # Save the tokenizers for evaluation
        self.text_tokenizer.save_pretrained(f'{run_config.output_path}/text_tokenizer')

        self.pdb_h5_path = os.path.join(run_config.data_path, 'pdb.h5')

        PretrainTask.__init__(self, run_config)

    def build_task_model(self):
        task_model = LlamaForCausalLM.from_pretrained(self.run_config.text_model_name)
        task_model.resize_token_embeddings(len(self.text_tokenizer), pad_to_multiple_of=8)
        task_model.gradient_checkpointing_enable({'use_reentrant': True})
        return task_model

    def build_trainer(self):
        return Trainer(
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            data_collator=DataCollatorForPureLLM(self.text_tokenizer, self.run_config.max_length,
                                                 self.pdb_h5_path, self.run_config.protein_token),
        )
