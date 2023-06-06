import pandas as pd
import torch
from datasets import load_dataset, DatasetDict, Dataset
from nltk.tokenize import sent_tokenize
import random
from tqdm import tqdm
from transformers import BertTokenizer


def construct_dataset(tok_name: str = 'bert-base-uncased'):
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
    for split in ["train", "test", "validation"]:
        sentences = []
        tokenized = []
        labels = []
        for sample in tqdm(dataset[split]):
            sample_sentences = sent_tokenize(sample["text"], language="english")
            # sorting out latex math env
            sample_sentences = [sample_sentence.strip('"') for sample_sentence in sample_sentences if "$" not in sample_sentence]
            sample_labels = [sample["label"] for i in range(len(sample_sentences))]
            sample_tokenized = [tokenizer.encode(sample_sentence, return_tensors='pt')[0] for sample_sentence in sample_sentences]
            sentences.extend(sample_sentences)
            labels.extend(sample_labels)
            tokenized.extend(sample_tokenized)
        random.shuffle(sentences)
        random.shuffle(labels)
        random.shuffle(tokenized)
        new_datasets[split] = Dataset.from_dict({"sent": sentences, "label": labels})
        new_datasets[split] = new_datasets[split].add_column("tokenized", [tok.numpy() for tok in tokenized])
        # new_datasets[split] = new_datasets[split].add_column("tokenized", tokenized)

    new_datasets.save_to_disk("../data/hm_dataset")


def load_dataset_info() -> DatasetDict:
    """
    Function for loading dataset from disc.
    1 --> machine generated
    0 --> human written text
    (model used: GPT3.5: GPT-3.5-turbo-0301)
    :return:
    """

    def to_tensor(example):
        example["tokenized"] = torch.tensor(example["tokenized"])
        return example

    dataset = DatasetDict.load_from_disk("../data/hm_dataset")
    dataset.set_format("torch")
    dataset.map(to_tensor)
    print("DATASET:")
    print(f'train-size: {len(dataset["train"])}')
    print(f'test-size: {len(dataset["test"])}')
    print(f'val-size: {len(dataset["validation"])}')

    print(dataset["train"][0])

    return dataset


#construct_dataset()
load_dataset_info()

