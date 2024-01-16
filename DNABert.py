import argparse
import os
import json

import numpy as np
import sklearn
import torch
from src.utils import (
    read_json,
    create_path,
    generate_UDir,
)
from src.dataset_utils import load_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    PreTrainedTokenizer,
    TrainingArguments,
)
from typing import Sequence, Dict, Optional
from dataclasses import dataclass, field

"""
This python script was 
"""


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluate the model",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    args = parser.parse_args()
    return args


@dataclass
class TrainingArgs(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=150, metadata={"help": "Maximum sequence length."}
    )
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=128)
    num_train_epochs: int = field(default=20)
    fp16: bool = field(default=False)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    weight_decay: float = field(default=1e-5)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    output_dir: str = field(default="output")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = (
        labels != -100
    )  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "auroc": sklearn.metrics.roc_auc_score(
            valid_labels, valid_predictions, average="macro"
        ),
        "auprc": sklearn.metrics.average_precision_score(
            valid_labels, valid_predictions, average="macro"
        ),
    }


"""
Compute metrics used for huggingface trainer.
"""


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DNABERT model")
    args = parse_arguments(parser)
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    config = read_json(json_path=args.json)
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=None,
        model_max_length=150,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    train_dataset, valid_dataset, test_dataset = load_datasets(
        batchSize=8,
        test_split=0.1,
        output_dir="data",
        tokenizer=tokenizer,
        return_loader=False,
    )

    Udir = generate_UDir(path=config.paths.get("results_path"))
    model_folder_path = os.path.join(config.paths.get("results_path"), Udir)
    create_path(model_folder_path)
    training_args = TrainingArgs()
    training_args.output_dir = os.path.join(model_folder_path, "output")
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=training_args.cache_dir,
        num_labels=2,
        trust_remote_code=True,
    )
    if args.train:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        # define trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_state()
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(model_folder_path, state_dict=cpu_state_dict)  # noqa

    if args.evaluate:
        results = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(model_folder_path, "eval_results.json"), "w") as f:
            json.dump(results, f)
