import math
from typing import Tuple

import pandas as pd
import torch
from datasets import load_dataset, DatasetDict, Dataset
from nltk.tokenize import sent_tokenize

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dependency_features import DependencyFeatureExtractor
from textscorer import ScorerModel
from functools import partial

"""
========================================================================================================================
BERT_DATASET
========================================================================================================================
"""


def construct_dataset_bert(tok_name: str = 'bert-base-uncased', max_len: int = 512):
    """
    Function for constructing dataset
    :return:
    """
    # Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(tok_name)
    # load private token
    with open("../data/token.txt", "r") as f:
        token = f.read().strip()

    # download private dataset from huggingface
    dataset = load_dataset("NicolaiSivesind/human-vs-machine", 'research_abstracts_labeled', use_auth_token=token)
    new_datasets = DatasetDict()
    # for split in ["test"]:
    for split in ["train", "test", "validation"]:
        sentences = []
        tokenized = []
        labels = []
        for sample in tqdm(dataset[split]):
            sample_sentences = sent_tokenize(sample["text"], language="english")
            # sorting out latex math env
            sample_sentences = [sample_sentence.strip('"') for sample_sentence in sample_sentences if
                                "$" not in sample_sentence]
            sample_labels = [sample["label"] for i in range(len(sample_sentences))]
            sample_tokenized = [tokenizer.encode_plus(sample_sentence,
                                                      add_special_tokens=True,
                                                      max_length=max_len,
                                                      return_token_type_ids=False,
                                                      # padding=True,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      truncation=True,
                                                      return_tensors='pt') for sample_sentence in sample_sentences]
            sentences.extend(sample_sentences)
            labels.extend(sample_labels)
            tokenized.extend(sample_tokenized)

        new_datasets[split] = Dataset.from_dict({"sent": sentences, "label": labels})
        new_datasets[split] = new_datasets[split].add_column("input_ids",
                                                             [tok['input_ids'].flatten().numpy() for tok in tokenized])
        new_datasets[split] = new_datasets[split].add_column("attention_mask",
                                                             [tok['attention_mask'].flatten().numpy() for tok in
                                                              tokenized])
        # new_datasets[split] = new_datasets[split].shuffle()

    new_datasets.save_to_disk(f"../data/hm_dataset_{tok_name.replace('/', '-')}")


def load_dataset_info_bert(do_print: bool = True, format_torch: bool = True,
                           tok_name: str = 'bert-base-uncased') -> DatasetDict:
    """
    Function for loading dataset from disc.
    1 --> machine generated
    0 --> human written text
    (model used: GPT3.5: GPT-3.5-turbo-0301)
    :return:
    """

    def to_tensor(example):
        example["input_ids"] = torch.tensor(example["input_ids"])
        example["attention_mask"] = torch.tensor(example["attention_mask"])
        # example["label"] = "human" if example["label"] == 0 else "machine"
        return example

    dataset = DatasetDict.load_from_disk(f"../data/hm_dataset_{tok_name.replace('/', '-')}")
    if format_torch:
        dataset.set_format("torch")
    # dataset = dataset.map(to_tensor)
    if do_print:
        print("DATASET:")
        print(f'train-size: {len(dataset["train"])}')
        print(f'test-size: {len(dataset["test"])}')
        print(f'val-size: {len(dataset["validation"])}')

    return dataset


"""
========================================================================================================================
FEATURE_DATASET
========================================================================================================================
"""


def construct_dataset_feat(feat_model: str = 'tree', funct: bool = False):
    """
    Function for constructing dataset
    :return:
    """

    def filter_nans(example):
        if sum(torch.isnan(torch.tensor(example["feat"]))) > 0:
            return False
        else:
            return True

    # use my model
    if "tree" in feat_model:
        model = DependencyFeatureExtractor(funct_analysis=funct)
    else:
        model = ScorerModel()

    # load private token
    with open("../data/token.txt", "r") as f:
        token = f.read().strip()

    # download private dataset from huggingface
    if "big_set" in feat_model:
        dataset = load_dataset("NicolaiSivesind/human-vs-machine", 'wiki_labeled', use_auth_token=token)
    else:
        dataset = load_dataset("NicolaiSivesind/human-vs-machine", 'research_abstracts_labeled', use_auth_token=token)
    new_datasets = DatasetDict()
    # for split in ["test"]:
    for split in ["train", "test", "validation"]:
        sentences = []
        feats = []
        labels = []
        for sample in tqdm(dataset[split]):
            sample_sentences = sent_tokenize(sample["text"], language="english")
            # sorting out latex math env
            sample_sentences = [sample_sentence.strip('"') for sample_sentence in sample_sentences if
                                "$" not in sample_sentence]
            sample_labels = [sample["label"] for i in range(len(sample_sentences))]
            sample_feats = [model(sample_sentence) for sample_sentence in sample_sentences]

            sentences.extend(sample_sentences)
            labels.extend(sample_labels)
            feats.extend(sample_feats)

        new_datasets[split] = Dataset.from_dict({"sent": sentences, "label": labels})
        new_datasets[split] = new_datasets[split].add_column("feat",
                                                             [feat.flatten().numpy() for feat in feats])

        # new_datasets[split] = new_datasets[split].shuffle()
    new_datasets = new_datasets.filter(filter_nans)
    new_datasets.save_to_disk(f"../data/hm_dataset_{feat_model}")


def load_dataset_info_feat(do_print: bool = True, format_torch: bool = True, feat_model: str = 'tree') -> DatasetDict:
    """
    Function for loading dataset from disc.
    1 --> machine generated
    0 --> human written text
    (model used: GPT3.5: GPT-3.5-turbo-0301)
    :return:
    """

    def find_nans(example):
        if sum(torch.isnan(example["feat"])) > 0:
            print(example)
        return example

    def filter_nans(example):
        if sum(torch.isnan(example["feat"])) > 0:
            return False
        else:
            return True

    dataset = DatasetDict.load_from_disk(f"../data/hm_dataset_{feat_model}")
    if format_torch:
        dataset.set_format("torch")
    # dataset = dataset.filter(filter_nans)
    if do_print:
        print("DATASET:")
        print(f'train-size: {len(dataset["train"])}')
        print(f'test-size: {len(dataset["test"])}')
        print(f'val-size: {len(dataset["validation"])}')

    return dataset


"""
========================================================================================================================
FEATURE_+_BERT_DATASET
========================================================================================================================
"""


def construct_dataset_bert_feat(tok_name: str = 'prajjwal1/bert-tiny', max_len: int = 512,
                                feat_model: str = 'tree_small'):
    """
    Function for constructing dataset
    :return:
    """

    # tokenizer
    def tok(example, tokenizer_obj: BertTokenizer, max_len: int = 512):
        tokenized = tokenizer_obj.encode_plus(example["sent"],
                                              add_special_tokens=True,
                                              max_length=max_len,
                                              return_token_type_ids=False,
                                              # padding=True,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              truncation=True,
                                              return_tensors='pt')
        example["input_ids"] = tokenized["input_ids"].flatten()
        example["attention_mask"] = tokenized["attention_mask"].flatten()
        return example

    # Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(tok_name)
    # filter function
    filter_tok = partial(tok, tokenizer_obj=tokenizer, max_len=max_len)
    # dataset
    dataset = DatasetDict.load_from_disk(f"../data/hm_dataset_{feat_model}")
    # train
    dataset["train"] = dataset["train"].add_column("input_ids", [0] * len(dataset["train"]))
    dataset["train"] = dataset["train"].add_column("attention_mask", [0] * len(dataset["train"]))
    # test
    dataset["test"] = dataset["test"].add_column("input_ids", [0] * len(dataset["test"]))
    dataset["test"] = dataset["test"].add_column("attention_mask", [0] * len(dataset["test"]))
    # val
    dataset["validation"] = dataset["validation"].add_column("input_ids", [0] * len(dataset["validation"]))
    dataset["validation"] = dataset["validation"].add_column("attention_mask", [0] * len(dataset["validation"]))

    dataset = dataset.map(filter_tok)

    dataset.save_to_disk(f"../data/hm_dataset_{tok_name.replace('/', '-')}+{feat_model}")


def load_dataset_info_bert_feat(do_print: bool = True,
                                format_torch: bool = True,
                                tok_name: str = 'prajjwal1/bert-tiny',
                                feat_model: str = 'tree_small') -> DatasetDict:
    """
    Function for loading dataset from disc.
    1 --> machine generated
    0 --> human written text
    (model used: GPT3.5: GPT-3.5-turbo-0301)
    :return:
    """

    def to_tensor(example):
        example["input_ids"] = torch.tensor(example["input_ids"])
        example["attention_mask"] = torch.tensor(example["attention_mask"])
        # example["label"] = "human" if example["label"] == 0 else "machine"
        return example

    dataset = DatasetDict.load_from_disk(f"../data/hm_dataset_{tok_name.replace('/', '-')}+{feat_model}")
    if format_torch:
        dataset.set_format("torch")
    # dataset = dataset.map(to_tensor)
    if do_print:
        print("DATASET:")
        print(f'train-size: {len(dataset["train"])}')
        print(f'test-size: {len(dataset["test"])}')
        print(f'val-size: {len(dataset["validation"])}')

        print(dataset["train"][0])
    return dataset


"""
========================================================================================================================
DataLoaders
========================================================================================================================
"""


def get_dataloaders(dataset: DatasetDict,
                    batch_size: int = 16,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Getting the Dataloaders.
    :param dataset:
    :param batch_size:
    :param num_workers:
    :return:
    """
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, num_workers=num_workers)
    print(len(train_loader))
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, val_loader


if __name__ == "__main__":
    construct_dataset_bert_feat()
    load_dataset_info_bert_feat()
