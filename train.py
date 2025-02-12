# uv run train.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "protobuf",
#     "transformers",
#     "datasets",
#     "wandb",
#     "accelerate>=0.26.0",
# ]
# ///

import wandb
import random
import glob
from datasets import Dataset
import torch
import transformers
import copy
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import json

transformers.logging.set_verbosity_info()
wandb.init(project="factorion", name="testing-algernon")

# Load tokenizer and model
model_name = "answerdotai/ModernBERT-base"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = ModernBertForMaskedLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=8192
    )

def augment_blueprint(
    blueprint,
    x_offset=None,
    y_offset=None,
    shuffle_entities=True,
    shuffle_icons=True,
):
    """Applies random augmentations to a Factorio blueprint JSON."""

    augmented = copy.deepcopy(blueprint)

    if shuffle_entities:
        random.shuffle(augmented["blueprint"]["entities"])

    if shuffle_icons:
        random.shuffle(augmented["blueprint"]["icons"])

    if x_offset is None:
        x_offset = random.randrange(-50, 50)

    if y_offset is None:
        y_offset = random.randrange(-50, 50)

    for entity in augmented["blueprint"]["entities"]:
        entity["position"]["x"] += x_offset
        entity["position"]["y"] += y_offset

    return augmented


def pretty_print_tokens(text, tokenizer):
    """Tokenize the given text, assign each token a random colour, then return
    the coloured text version of the tokens. Useful for visualising what the
    tokenizer is doing. Replaces the weird Ġ character with a space ' '."""

    colors = ['\033[31m', '\033[34m', '\033[32m', '\033[33m', '\033[35m', '\033[36m', '\033[37m', '\033[91m', '\033[92m', '\033[93m']
    colors_bg = ['\033[41m', '\033[44m', '\033[42m', '\033[43m', '\033[45m', '\033[46m', '\033[47m', '\033[91m', '\033[92m', '\033[93m']
    reset = '\033[0m'
    tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    colored_tokens = []
    last_color = None
    for token in tokens.numpy()[0]:
        color, color_bg = random.choice(list(zip(colors, colors_bg)))
        while color == last_color:
            color = random.choice(colors)
        last_color = color
        string = tokenizer.convert_ids_to_tokens([token])[0].replace("Ġ", f"{reset}{color_bg} {reset}{color}")
        colored_tokens.append(f"{color}{string}{reset}")
    return ''.join(colored_tokens)


def prepare_dataset(tokenizer, glob_str="blueprints/*.json", augmentations=50):
    """Prepares dataset by tokenizing and augmenting multiple times."""
    paths = glob.glob(glob_str)
    blueprints = []
    for path in paths:
        with open(path, 'r') as f:
            blueprints.append(json.load(f))

    augmented_blueprints = []
    for _ in range(augmentations):
        for blueprint in blueprints:
            augmented_blueprints.append(augment_blueprint(blueprint))

    # Note: masking not done here, masking needs to be done during the training
    # process
    stringified_blueprints = [
        json.dumps(bp, separators=(',', ':'))
        for bp in (blueprints + augmented_blueprints)
    ]

    dataset = Dataset.from_dict({"text": stringified_blueprints}).map(tokenize_function, batched=True, remove_columns=["text"])

    split_dataset = dataset.train_test_split(test_size=0.25, seed=42)

    return split_dataset['train'], split_dataset['test']


tokenized_trn, tokenized_val = prepare_dataset(tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

output_dir = "./modernbert-ft-factorio"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=1,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    remove_unused_columns=False,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_trn,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
# Left commented because it's handy sometimes: resume training from a checkpoint
# trainer.train(resume_from_checkpoint=True)

trainer.save_model(output_dir + '-saved-model')
