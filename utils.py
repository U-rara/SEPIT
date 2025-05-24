import dataclasses
import os.path
import random
from enum import Enum, auto
from typing import List, Dict

import h5py
import pandas as pd
import torch
from torch import nn
from transformers.data.data_collator import DataCollatorMixin


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def torch_mask_tokens(inputs, tokenizer, mlm_probability, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 100% MASK.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


class DataCollatorForProteinTextCLIPPretrain(DataCollatorMixin):
    def __init__(
        self,
        protein_tokenizer,
        text_tokenizer,
        pdb_h5_file,
        sequence_only,
        mlm_probability,
    ):
        self.return_tensors = "pt"
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.pdb_h5_file = pdb_h5_file
        self.sequence_only = sequence_only
        self.mlm_probability = mlm_probability

    def torch_call(self, examples):
        sequences = []
        if not self.sequence_only:
            node_positions = []
        with h5py.File(self.pdb_h5_file, "r") as pdb_h5:
            for e in examples:
                alphafold_id = e["id"]
                group = pdb_h5[alphafold_id]
                sequence = group["sequence"][()][0].decode()
                node_position = group["node_position"][()]
                sequences.append(sequence)
                if not self.sequence_only:
                    node_positions.append(torch.from_numpy(node_position)[: 1024 - 2])

        protein_tokenized = self.protein_tokenizer(
            sequences,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        text_tokenized = self.text_tokenizer(
            [e["caption"] for e in examples],
            truncation=True,
            max_length=512,  # 256->miss 10% ， 512-> miss 1%
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        if not self.sequence_only:
            node_position = torch.zeros(
                (len(examples), protein_tokenized["input_ids"].shape[-1], 3)
            )
            for i, e in enumerate(examples):
                node_position[i, 1 : node_positions[i].shape[0] + 1, :] = (
                    node_positions[i]
                )

        if self.mlm_probability == 0.0:
            if not self.sequence_only:
                return {
                    "protein_input_ids": protein_tokenized["input_ids"],
                    "protein_attention_mask": protein_tokenized["attention_mask"],
                    "text_input_ids": text_tokenized["input_ids"],
                    "text_attention_mask": text_tokenized["attention_mask"],
                    "node_position": node_position,
                }
            else:
                return {
                    "protein_input_ids": protein_tokenized["input_ids"],
                    "protein_attention_mask": protein_tokenized["attention_mask"],
                    "text_input_ids": text_tokenized["input_ids"],
                    "text_attention_mask": text_tokenized["attention_mask"],
                }
        else:
            protein_masked_input_ids, protein_masked_labels = torch_mask_tokens(
                protein_tokenized["input_ids"].clone(),
                self.protein_tokenizer,
                self.mlm_probability,
            )
            if not self.sequence_only:
                return {
                    "protein_input_ids": protein_tokenized["input_ids"],
                    "protein_attention_mask": protein_tokenized["attention_mask"],
                    "text_input_ids": text_tokenized["input_ids"],
                    "text_attention_mask": text_tokenized["attention_mask"],
                    "node_position": node_position,
                    "protein_masked_input_ids": protein_masked_input_ids,
                    "protein_masked_labels": protein_masked_labels,
                }
            else:
                return {
                    "protein_input_ids": protein_tokenized["input_ids"],
                    "protein_attention_mask": protein_tokenized["attention_mask"],
                    "text_input_ids": text_tokenized["input_ids"],
                    "text_attention_mask": text_tokenized["attention_mask"],
                    "protein_masked_input_ids": protein_masked_input_ids,
                    "protein_masked_labels": protein_masked_labels,
                }


class DataCollatorForPIT(DataCollatorMixin):
    def __init__(
        self,
        protein_tokenizer,
        text_tokenizer,
        max_length,
        pdb_h5_file,
        protein_token,
        sequence_only,
        drop_structure_rate=0.0,
    ):
        self.return_tensors = "pt"
        self.max_length = max_length
        self.pdb_h5_file = pdb_h5_file
        self.protein_token = protein_token
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.sequence_only = sequence_only
        self.drop_structure_rate = drop_structure_rate
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(
            self.protein_token
        )
        self.protein_sequence_dataframe = pd.read_csv(
            os.path.join(os.path.dirname(pdb_h5_file), "sequences.csv")
        )
        self.protein_sequence_dataframe.set_index("id", inplace=True)
        assert isinstance(
            self.protein_token_id, int
        ), "Protein token not found in tokenizer"

    def torch_call(self, examples):
        conversations = []
        sequences = []
        if not self.sequence_only:
            node_positions = []
        with h5py.File(self.pdb_h5_file, "r") as pdb_h5:
            for e in examples:
                id = e["id"]
                conversation = e["conversation"]

                if id in pdb_h5:
                    group = pdb_h5[id]
                    sequence = group["sequence"][()][0].decode()
                    node_position = group["node_position"][()]
                    for conv in conversation:
                        conv["content"] = conv["content"].replace(
                            self.protein_token, "<bop>" + self.protein_token + "<eop>"
                        )
                    if not self.sequence_only:
                        node_positions.append(
                            torch.from_numpy(node_position)[: self.max_length - 2]
                        )
                else:
                    sequence = self.protein_sequence_dataframe.loc[id].values[0]
                    node_positions.append(torch.zeros((self.max_length - 2, 3)))

                conversations.append(conversation)
                sequences.append(sequence)

        protein_inputs = self.protein_tokenizer(
            sequences,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        text_inputs = preprocess_llama_2(
            conversations, self.text_tokenizer, self.max_length
        )

        if not self.sequence_only:
            node_position = torch.zeros(
                (len(examples), protein_inputs["input_ids"].shape[-1], 3)
            )
            for i, e in enumerate(examples):
                if random.random() > self.drop_structure_rate:
                    node_position[i, 1 : node_positions[i].shape[0] + 1, :] = (
                        node_positions[i]
                    )

            return {
                "protein_input_ids": protein_inputs["input_ids"],
                "protein_attention_mask": protein_inputs["attention_mask"],
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "labels": text_inputs["labels"],
                "node_position": node_position,
            }
        else:
            return {
                "protein_input_ids": protein_inputs["input_ids"],
                "protein_attention_mask": protein_inputs["attention_mask"],
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "labels": text_inputs["labels"],
            }


class DataCollatorForPureLLM(DataCollatorMixin):
    def __init__(self, text_tokenizer, max_length, pdb_h5_file, protein_token):
        self.return_tensors = "pt"
        self.max_length = max_length
        self.pdb_h5_file = pdb_h5_file
        self.protein_token = protein_token
        self.text_tokenizer = text_tokenizer
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(
            self.protein_token
        )
        assert isinstance(
            self.protein_token_id, int
        ), "Protein token not found in tokenizer"

    def torch_call(self, examples):
        conversations = []
        with h5py.File(self.pdb_h5_file, "r") as pdb_h5:
            for e in examples:
                id = e["id"]
                conversation = e["conversation"]
                group = pdb_h5[id]
                sequence = group["sequence"][()][0].decode()[: self.max_length]
                for conv in conversation:
                    conv["content"] = conv["content"].replace(
                        self.protein_token, "<bop>" + sequence + "<eop>"
                    )
                conversations.append(conversation)

        text_inputs = preprocess_llama_2(
            conversations, self.text_tokenizer, self.max_length * 2
        )

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["labels"],
        }


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


def preprocess_llama_2(sources, tokenizer, max_length) -> Dict:
    conv = get_conv()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]

    targets = input_ids.clone()

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = -100
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = -100

            cur_len += round_len
        target[cur_len:] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": targets,
    }


def get_conv():
    conv = Conversation(
        system="You are a helpful biological assistant. "
        "You are able to understand the protein that the user provides, "
        "and assist the user with a variety of tasks using natural language.",
        roles=["USER", "ASSISTANT"],
        version="llama_v2",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_2,
        sep="<s>",
        sep2="</s>",
    )
    return conv


class HookTool:
    def __init__(self):
        self.feature = []

    def hook_function(self, module, feature_in, feature_out):
        self.feature.append(feature_out.detach().cpu())


def get_gating_logit_by_hook(model):
    feature_hooks = []
    for n, m in model.named_modules():
        if "mlp.gate" in n and isinstance(m, nn.Linear):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_function)
            feature_hooks.append(cur_hook)
    return feature_hooks
