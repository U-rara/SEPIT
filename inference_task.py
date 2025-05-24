import json
import os.path
from copy import deepcopy

import h5py
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import preprocess_llama_2, get_gating_logit_by_hook


class InferenceTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.protein_tokenizer, self.text_tokenizer, self.model = (
            self.load_tokenizer_and_model()
        )
        self.dataset, self.pdb_h5_path = self.build_dataset()

    def load_tokenizer_and_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class PITStage2InferenceTask(InferenceTask):
    def __init__(self, run_config):
        super().__init__(run_config)
        self.rank = self.run_config.rank
        self.world_size = self.run_config.world_size
        self.device = torch.device(
            "cuda"
        )  # use CUDA_VISIBLE_DEVICES to control the device
        if self.rank == -1:
            assert self.world_size == 1
            self.output_file = os.path.join(
                self.run_config.output_path, f"{self.run_config.dataset}_output.jsonl"
            )
        else:
            self.output_file = (
                os.path.join(
                    self.run_config.output_path,
                    f"{self.run_config.dataset}_output.jsonl",
                )
                + f".{self.rank}"
            )
        torch.set_grad_enabled(False)
        self.model.half().to(self.device).eval()

    def load_tokenizer_and_model(self):
        model_name = self.run_config.model_name
        protein_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, "protein_tokenizer")
        )

        text_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, "text_tokenizer")
        )
        text_tokenizer.padding_side = "left"

        from model.pit.modeling_pit import EvalPITForConditionGeneration

        model = EvalPITForConditionGeneration.from_pretrained(model_name)
        if (
            hasattr(self.run_config, "return_gating_logit")
            and self.run_config.return_gating_logit is not None
        ):
            self.feature_hooks = get_gating_logit_by_hook(model)
            self.all_gating_logits = {}
        return protein_tokenizer, text_tokenizer, model

    def build_dataset(self):
        pdb_h5_path = os.path.join(self.run_config.data_path, "pdb.h5")
        dataset = load_dataset(
            "json",
            data_files={
                self.run_config.subset: os.path.join(
                    self.run_config.data_path,
                    self.run_config.dataset,
                    f"{self.run_config.subset}.jsonl",
                ),
            },
        )
        return dataset, pdb_h5_path

    def batch_inference(self, batch):
        ids, conversations = batch["id"], batch["conversation"]
        origin_conversations = deepcopy(conversations)
        prompts = []
        for prompt in conversations:
            prompt[0]["content"] = prompt[0]["content"].replace(
                "<protein>", "<bop><protein><eop>"
            )
            prompt[1]["content"] = None
            prompts.append(prompt)

        sequences = []
        if "sequence" in batch:
            sequences += batch["sequence"]
        else:
            if not self.run_config.sequence_only:
                node_positions = []
            with h5py.File(self.pdb_h5_path, "r") as pdb_h5:
                for id in ids:
                    group = pdb_h5[id]
                    sequences.append(group["sequence"][()][0].decode())
                    if not self.run_config.sequence_only:
                        node_positions.append(group["node_position"][()][: 1024 - 2])

        inputs = preprocess_llama_2(prompts, self.text_tokenizer, 256)
        inputs.pop("labels")
        protein_tokenized = self.protein_tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        inputs["protein_input_ids"] = protein_tokenized["input_ids"]
        inputs["protein_attention_mask"] = protein_tokenized["attention_mask"]
        inputs["node_position"] = torch.zeros((len(batch["id"]), 1024, 3))
        if not self.run_config.sequence_only:
            for i, node_position in enumerate(node_positions):
                inputs["node_position"][i, 1 : node_position.shape[0] + 1, :] = (
                    torch.from_numpy(node_position)
                )

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024 - 256,
            pad_token_id=self.text_tokenizer.pad_token_id,
        )

        if (
            hasattr(self.run_config, "return_gating_logit")
            and self.run_config.return_gating_logit is not None
        ):
            self.all_gating_logits[len(self.all_gating_logits)] = dict(
                gating_logit=[i.feature for i in self.feature_hooks],
                proteins=inputs["protein_input_ids"],
                input_ids=inputs["input_ids"],
                output_ids=outputs,
            )
            for i in range(len(self.feature_hooks)):
                self.feature_hooks[i].feature = []
            print("The number of hooks is:", len(self.feature_hooks))

        responses = self.text_tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return ids, origin_conversations, responses

    def run(self):
        if self.rank == -1:  # single process
            worker_dataset = self.dataset[self.run_config.subset]
        else:
            worker_dataset = self.dataset[self.run_config.subset].shard(
                num_shards=self.world_size, index=self.rank
            )

        batch_size = self.run_config.batch_size
        dataset_size = len(worker_dataset)
        for i in tqdm(range(0, dataset_size, batch_size)):
            batch = worker_dataset[i : min(i + batch_size, dataset_size)]
            ids, conversations, responses = self.batch_inference(batch)
            for id, conversation, response in zip(ids, conversations, responses):
                output_dict = {
                    "id": id,
                    "conversation": conversation,
                    "response": response,
                }
                with open(self.output_file, "a") as f:
                    f.write(json.dumps(output_dict))
                    f.write("\n")
        if (
            hasattr(self.run_config, "return_gating_logit")
            and self.run_config.return_gating_logit is not None
        ):
            torch.save(
                self.all_gating_logits, f"{self.run_config.return_gating_logit}.pt"
            )
