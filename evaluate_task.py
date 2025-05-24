import json
import os.path
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from bert_score import BERTScorer
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


class EvaluateTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.dataset = self.build_data()

    def build_data(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class QAEvaluateTask(EvaluateTask):
    def __init__(self, run_config):
        if hasattr(run_config, "id_list"):
            self.id_list = pd.read_csv(run_config.id_list)["id"].tolist()
        super().__init__(run_config)
        self.set_scorer()

    def build_data(self):
        dataset = load_dataset("json", data_files=self.run_config.data_path)
        if hasattr(self, "id_list"):
            dataset["train"] = dataset["train"].filter(
                lambda x: x["id"] in self.id_list
            )
        return dataset

    def get_answer_and_output(self, data):
        id = data["id"]
        conversation = data["conversation"]
        response = data["response"]

        question = conversation[0]["content"]
        answer = conversation[1]["content"]

        pattern = re.compile(question.replace("<protein>", "([A-Za-z0-9\s/\-_]*?)"))
        match = pattern.search(response)
        if match:
            output = response[match.end() :].strip()
        else:
            output = response.strip()
        output = output.replace("? ", "").strip()
        return id, answer, output

    def set_scorer(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bert_config = AutoConfig.from_pretrained(self.run_config.bert_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.run_config.bert_model_name
        )
        self.bert_score_scorer = BERTScorer(
            model_type=self.run_config.bert_model_name,
            num_layers=self.bert_config.num_hidden_layers,
        )

    def calculate_scores(self, data):
        id, answer, output = self.get_answer_and_output(data)
        cands_tokenized = self.bert_tokenizer.tokenize(output, add_special_tokens=False)
        refs_tokenized = self.bert_tokenizer.tokenize(answer, add_special_tokens=False)

        bleu1 = corpus_bleu([[refs_tokenized]], [cands_tokenized], weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(
            [[refs_tokenized]], [cands_tokenized], weights=(0.5, 0.5, 0, 0)
        )
        bleu3 = corpus_bleu(
            [[refs_tokenized]], [cands_tokenized], weights=(0.33, 0.33, 0.33, 0)
        )
        bleu4 = corpus_bleu(
            [[refs_tokenized]], [cands_tokenized], weights=(0.25, 0.25, 0.25, 0.25)
        )

        rouge = self.rouge_scorer.score(output, answer)

        meteor = meteor_score([refs_tokenized], cands_tokenized)

        P, R, F1 = self.bert_score_scorer.score(
            [
                self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(
                        output,
                        max_length=500,
                        truncation=True,
                        add_special_tokens=False,
                    )
                )
            ],
            [
                self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(
                        answer,
                        max_length=500,
                        truncation=True,
                        add_special_tokens=False,
                    )
                )
            ],
        )
        scores = {
            "id": id,
            "answer": answer,
            "output": output,
            "bleu1": bleu1 * 100,
            "bleu2": bleu2 * 100,
            "bleu3": bleu3 * 100,
            "bleu4": bleu4 * 100,
            "rouge1": rouge["rouge1"].fmeasure * 100,
            "rouge2": rouge["rouge2"].fmeasure * 100,
            "rougeL": rouge["rougeL"].fmeasure * 100,
            "meteor": meteor * 100,
            "bertscore_p": P.mean().item() * 100,
            "bertscore_r": R.mean().item() * 100,
            "bertscore_f1": F1.mean().item() * 100,
        }
        return scores

    def run(self):
        avg_scores = defaultdict(list)

        pbar = tqdm(total=len(self.dataset["train"]), desc="Evaluating", unit="sample")

        def process_data(data):
            scores = self.calculate_scores(data)
            pbar.update(1)
            return scores

        with ThreadPoolExecutor(max_workers=1) as executor:  # NLTK is not thread safe
            results = list(executor.map(process_data, self.dataset["train"]))

        pbar.close()

        all_records = results

        for scores in results:
            for key in scores:
                if key not in ["id", "answer", "output"]:  # Skip non-numeric fields
                    avg_scores[key].append(scores[key])

        averages = {
            key: sum(values) / len(values) for key, values in avg_scores.items()
        }

        output_dict = {"average_scores": averages, "individual_records": all_records}

        print("Average scores:", averages)
        with open(os.path.join(self.run_config.output_path, "metrics.json"), "w") as f:
            json.dump(output_dict, f, indent=2)


class APIModelQAEvaluateTask(QAEvaluateTask):
    def get_answer_and_output(self, data):
        id = data["id"]
        conversation = data["conversation"]
        response = data["response"]
        answer = conversation[1]["content"]
        return id, answer, response


class YNEvaluateTask(QAEvaluateTask):
    def __init__(self, run_config):
        if hasattr(run_config, "id_list"):
            self.id_list = pd.read_csv(run_config.id_list)["id"].tolist()
        EvaluateTask.__init__(self, run_config)

    def get_answer_and_output(self, data):
        id = data["id"]
        conversation = data["conversation"]
        response = data["response"]

        answer = conversation[1]["content"].lower()

        output = (
            re.findall(r"\byes\b|\bno\b", response, re.IGNORECASE)[-1]
            if (re.search(r"\byes\b|\bno\b", response, re.IGNORECASE))
            else None
        )
        return id, answer, output.lower() if output else "None"

    def calculate_scores(self, data):
        id, answer, output = self.get_answer_and_output(data)
        scores = {"accuracy": float(answer == output)}  # or output == 'None'
        return scores


APIModelYNEvaluateTask = YNEvaluateTask
