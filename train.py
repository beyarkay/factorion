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
from datasets import load_dataset, Dataset
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

dataset = load_dataset("text", data_files="blueprints/*.json")
split_dataset = dataset["train"].train_test_split(test_size=0.25)

trn_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

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
    upgrade_entities_prob=0.5
):
    """Applies random augmentations to a Factorio blueprint JSON."""
    ENTITY_REPLACEMENTS = {
        "small-electric-pole":       "medium-electric-pole",
        "transport-belt":            "fast-transport-belt",
        "fast-transport-belt":       "express-transport-belt",
        "express-transport-belt":    "turbo-transport-belt",
        "inserter":                  "fast-inserter",
        # Don't do the undergrounds just yet, because we'd have to swap *both* of them
        # "underground-belt":          "fast-underground-belt",
        # "fast-underground-belt":     "express-underground-belt",
        # "express-underground-belt":  "turbo-underground-belt",
        "splitter":                  "fast-splitter",
        "fast-splitter":             "express-splitter",
        "express-splitter":          "turbo-splitter",
    }

    augmented = copy.deepcopy(blueprint)

    if shuffle_entities:
        random.shuffle(augmented["blueprint"]["entities"])

    if shuffle_icons:
        random.shuffle(augmented["blueprint"]["icons"])

    if x_offset is None:
        x_offset = random.randrange(-10, 10)

    if y_offset is None:
        y_offset = random.randrange(-10, 10)

    for entity in augmented["blueprint"]["entities"]:
        entity["position"]["x"] += x_offset
        entity["position"]["y"] += y_offset
        if entity["name"] in ENTITY_REPLACEMENTS and random.random() < upgrade_entities_prob:
            entity["name"] = ENTITY_REPLACEMENTS[entity["name"]]

    return augmented


def mask_blueprint(blueprint_dict, mask_prob=0.1):
    """Mask out specific fields in the blueprint JSON."""
    masked = blueprint_dict.copy()

    for entity in masked["blueprint"]["entities"]:
        if random.random() <= mask_prob:
            entity["name"] = "[MASK]"

        if random.random() <= mask_prob:
            entity["position"]["x"] = "[MASK]"

        if random.random() <= mask_prob:
            entity["position"]["y"] = "[MASK]"

        if "direction" in entity and random.random() <= mask_prob:
            entity["type"] = "[MASK]"

        if "type" in entity and random.random() <= mask_prob:
            entity["type"] = "[MASK]"

    return masked


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
        colored_tokens.append(f"{color}{string}{reset}")  # Color the token
    return ''.join(colored_tokens)


def prepare_dataset(tokenizer, glob_str="blueprints/*.json", augmentations=0):
    """Prepares dataset by tokenizing and augmenting multiple times."""
    paths = glob.glob(glob_str)
    blueprints = []
    for path in paths:
        with open(path, 'r') as f:
            blueprints.append(json.load(f))

    augmented_blueprints = []
    for _ in range(augmentations):
        for blueprint in blueprints:
            augmented_blueprints.append(augment_blueprint(
                blueprint,
                upgrade_entities_prob=0
            ))

    # Note: masking not done here, masking needs to be done during the training
    # process
    stringified_blueprints = [
        json.dumps(bp, separators=(',', ':'))
        for bp in (blueprints + augmented_blueprints)
    ]

    dataset = Dataset.from_dict({"text": stringified_blueprints}).map(tokenize_function, batched=True, remove_columns=["text"])

    # Don't tokenize just yet, that happens in the collator
    # full_dataset = dataset.map(
    #     lambda x: tokenize_function(x, tokenizer),
    #     batched=True,
    #     remove_columns=["text"]
    # )

    split_dataset = dataset.train_test_split(test_size=0.25, seed=42)

    return split_dataset['train'], split_dataset['test']

class DataCollatorForSelectiveMasking(DataCollatorForLanguageModeling):
    """A custom collator so that we can selectively mask parts of the JSON
    blueprint that we actually care about."""
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer, mlm=True, mlm_probability=mlm_probability)

    def mask_json(self, json_data):
        """Mask specific fields in the JSON before tokenization."""
        masked_json = json_data.copy()
        if "blueprint" in masked_json and "entities" in masked_json["blueprint"]:
            for entity in masked_json["blueprint"]["entities"]:
                if random.random() < self.mlm_probability:
                    entity["name"] = "[MASK]"
                if random.random() < self.mlm_probability:
                    entity["position"]["x"] = "[MASK]"
                if random.random() < self.mlm_probability:
                    entity["position"]["y"] = "[MASK]"
                if "type" in entity and random.random() < self.mlm_probability:
                    entity["type"] = "[MASK]"
        return masked_json

    def __call__(self, examples):
        """Apply masking before tokenization."""
        masked_examples = [self.mask_json(ex) for ex in examples]
        # tokenized = [self.tokenizer(json.dumps(ex), truncation=True, padding="max_length") for ex in masked_examples]
        tokenized = self.tokenizer([json.dumps(ex) for ex in masked_examples], truncation=True, padding="max_length")
        # print('type of data: ', type(data))
        # print('data is: ', data)
        # return data
        return super().__call__(tokenized)

# Use the custom data collator
# data_collator = DataCollatorForSelectiveMasking(tokenizer, mlm_probability=0.15)

tokenized_trn, tokenized_val = prepare_dataset(tokenizer)
# print(tokenized_trn)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.05
)

output_dir = "./modernbert-ft-factorio"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
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
