from collections import defaultdict
from functools import partial
from typing import Optional

from datasets import DatasetDict, Dataset, load_dataset
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from dependency_features import DependencyFeatureExtractor
from detector_models import HumanMachineClassifierBertTiny, HumanMachineClassifierBertTinyFeatureTiny
from extract_dataset import load_dataset_info_bert, get_dataloaders, construct_dataset_bert, load_dataset_info_feat, \
    construct_dataset_feat, load_dataset_info_bert_feat
from paraphraser import Paraphraser, AutoParaphraser
from textscorer import ScorerModel

TOKENIZERS_PARALLELISM = True

class Experiment:
    def __init__(self,
                 model: nn.Module,
                 model_path: str,
                 paraphraser: Paraphraser,
                 batch_size: int = 32,
                 max_len: int = 512,
                 device: str = "cuda:0",
                 feat_model: str = "tree_small",
                 tok_name: str = "prajjwal1/bert-tiny"
                 ):
        # load model
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        # Bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tok_name)

        # paraphraser
        self.paraphraser = paraphraser

        # feat model
        self.dep_feat_model = DependencyFeatureExtractor(funct_analysis=False)

        # load dataset
        if self.device != "cpu":
            format_torch = True
        else:
            format_torch = False

        dataset = load_dataset_info_bert_feat(do_print=False,
                                              format_torch=format_torch,
                                              tok_name=tok_name,
                                              feat_model=feat_model)

        _, tel, _ = get_dataloaders(dataset, batch_size=batch_size)
        self.test_data_loader = tel
        self.batch_size = batch_size
        self.max_len = max_len
        self.feat_model = feat_model
        self.tok_name = tok_name

    @staticmethod
    def paraphrase_dataset(dataset: Dataset,
                           paraphraser: Paraphraser,
                           tokenizer: BertTokenizer,
                           dep_feat_model: DependencyFeatureExtractor,
                           max_len: int = 512):
        # tokenizer
        def tok(example,
                tokenizer_obj: BertTokenizer,
                paraphraser_obj: Paraphraser,
                dep_feat_model_obj: DependencyFeatureExtractor,
                max_len: int = 512):
            # print(example["sent"])
            example["sent"] = paraphraser_obj(example["sent"])[0]
            tokenized = tokenizer_obj.encode_plus(example["sent"],
                                                  add_special_tokens=True,
                                                  max_length=max_len,
                                                  return_token_type_ids=False,
                                                  # padding=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  truncation=True,
                                                  return_tensors='pt')
            example["input_ids"] = tokenized["input_ids"].flatten().numpy()
            example["attention_mask"] = tokenized["attention_mask"].flatten().numpy()
            # print(example["sent"])
            example["feat"] = dep_feat_model_obj(example["sent"]).flatten().numpy()
            return example

        map_tok_feat = partial(tok,
                               tokenizer_obj=tokenizer,
                               paraphraser_obj=paraphraser,
                               dep_feat_model_obj=dep_feat_model,
                               max_len=max_len)

        # nan filter
        def filter_nans(example):
            if sum(torch.isnan(torch.tensor(example["feat"]))) > 0:
                return False
            else:
                return True

        dataset = dataset.map(map_tok_feat)
        dataset = dataset.filter(filter_nans)
        return dataset

    def get_predictions(self, data_loader: DataLoader):
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        with torch.no_grad():
            for d in tqdm(data_loader, desc="test"):
                texts = d["sent"]
                input_feats = d["feat"].to(self.device)
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label"].float().to(self.device)
                preds = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feat=input_feats
                ).flatten()
                review_texts.extend(texts)
                predictions.extend(preds.round())
                prediction_probs.extend(preds)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values

    def test(self):
        # test_acc, _ = self.test_model()
        _, y_pred, y_pred_probs, y_test = self.get_predictions(self.test_data_loader)
        # print(test_acc)
        print(classification_report(y_test, y_pred, target_names=["human", "machine"]))

    def test_paraphrase(self, range_: Optional[int] = None):
        dataset_test: Dataset = DatasetDict.load_from_disk(f"../data/hm_dataset_{self.tok_name.replace('/', '-')}+{self.feat_model}")["test"]
        if range_ is not None:
            dataset_test = dataset_test.select(range(range_))
        paraphrased_dataset = self.paraphrase_dataset(dataset_test,
                                                      self.paraphraser,
                                                      self.tokenizer,
                                                      self.dep_feat_model,
                                                      self.max_len)
        if self.device != "cpu":
            paraphrased_dataset.set_format("torch")

        test_loader = DataLoader(paraphrased_dataset, batch_size=self.batch_size, num_workers=4)

        _, y_pred, y_pred_probs, y_test = self.get_predictions(test_loader)
        print(classification_report(y_test, y_pred, target_names=["human", "machine"]))


def dep_test():
    # dep test
    m_dep = HumanMachineClassifierBertTiny()
    m_dep_path = "../data/feat_exp/best_model_state.bin"
    para = AutoParaphraser("humarin/chatgpt_paraphraser_on_T5_base", cuda=True)
    feat_model = "tree_small"
    tok_name = "prajjwal1/bert-tiny"
    exp_dep = Experiment(model=m_dep,
                         model_path=m_dep_path,
                         paraphraser=para,
                         feat_model=feat_model,
                         tok_name=tok_name
                         )
    #exp_dep.test()
    print(50 * "=")
    exp_dep.test_paraphrase(100)

def dep_feat_test():
    # dep test
    m_dep_feat = HumanMachineClassifierBertTinyFeatureTiny()
    m_dep_feat_path = "../data/feat_dep_exp/best_model_state.bin"
    para = AutoParaphraser("humarin/chatgpt_paraphraser_on_T5_base", cuda=True)
    feat_model = "tree_small"
    tok_name = "prajjwal1/bert-tiny"
    exp_dep = Experiment(model=m_dep_feat,
                         model_path=m_dep_feat_path,
                         paraphraser=para,
                         feat_model=feat_model,
                         tok_name=tok_name
                         )
    #exp_dep.test()
    print(50 * "=")
    exp_dep.test_paraphrase(100)


if __name__ == "__main__":
    dep_test()
    dep_feat_test()
