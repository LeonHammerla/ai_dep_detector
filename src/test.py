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
from detector_models import HumanMachineClassifierBertTiny, HumanMachineClassifierBertTinyFeatureTiny, \
    HumanMachineClassifierFeature5
from extract_dataset import load_dataset_info_bert, get_dataloaders, construct_dataset_bert, load_dataset_info_feat, \
    construct_dataset_feat, load_dataset_info_bert_feat, dep_parse_spacy_dataset
from paraphraser import Paraphraser, AutoParaphraser
from textscorer import ScorerModel

TOKENIZERS_PARALLELISM = True


class Experiment:
    def __init__(self,
                 model: nn.Module,
                 model_path: str,
                 # paraphraser: Paraphraser,
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
        # self.paraphraser = paraphraser

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

    def paraphrase_ds(self, paraphraser: Paraphraser, range_: Optional[int] = None,
                      save_load: str = "save") -> DataLoader:
        if save_load == "save":
            dataset_test: Dataset = \
                DatasetDict.load_from_disk(f"../data/hm_dataset_{self.tok_name.replace('/', '-')}+{self.feat_model}")[
                    "test"]
            if range_ is not None:
                dataset_test = dataset_test.select(range(range_))
            paraphrased_dataset = self.paraphrase_dataset(dataset_test,
                                                          paraphraser,
                                                          self.tokenizer,
                                                          self.dep_feat_model,
                                                          self.max_len)
            paraphrased_dataset.save_to_disk(f"../data/paraphrased_ds/para_set")
        elif save_load == "load":
            paraphrased_dataset = Dataset.load_from_disk(f"../data/paraphrased_ds/para_set")
        else:
            raise Exception

        if self.device != "cpu":
            paraphrased_dataset.set_format("torch")

        test_loader = DataLoader(paraphrased_dataset, batch_size=self.batch_size, num_workers=4)
        return test_loader

    def test(self, test_loader: DataLoader) -> str:
        _, y_pred, y_pred_probs, y_test = self.get_predictions(test_loader)
        return classification_report(y_test, y_pred, target_names=["human", "machine"])


class Experiment2:
    def __init__(self,
                 model: nn.Module,
                 model_path: str,
                 ds_before: Dataset,
                 ds_after: Dataset,
                 batch_size: int = 32,
                 max_len: int = 512,
                 device: str = "cuda:0",
                 ):
        # load model
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.ds_before = ds_before
        self.ds_after = ds_after

        # load dataset
        if self.device != "cpu":
            self.ds_before.set_format("torch")
            self.ds_after.set_format("torch")

        self.test_data_loader_before = DataLoader(self.ds_before, batch_size=batch_size, num_workers=4)
        self.test_data_loader_after = DataLoader(self.ds_after, batch_size=batch_size, num_workers=4)
        self.batch_size = batch_size
        self.max_len = max_len

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

    def test(self, test_loader: DataLoader) -> str:
        _, y_pred, y_pred_probs, y_test = self.get_predictions(test_loader)
        return classification_report(y_test, y_pred, target_names=["human", "machine"])


class Experiment3:
    def __init__(self,
                 model: nn.Module,
                 model_path: str,
                 ds_before: Dataset,
                 ds_after: Dataset,
                 batch_size: int = 32,
                 max_len: int = 512,
                 device: str = "cuda:0",
                 ):
        # load model
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.ds_before = ds_before
        self.ds_after = ds_after

        # load dataset
        if self.device != "cpu":
            self.ds_before.set_format("torch")
            self.ds_after.set_format("torch")

        self.test_data_loader_before = DataLoader(self.ds_before, batch_size=batch_size, num_workers=4)
        self.test_data_loader_after = DataLoader(self.ds_after, batch_size=batch_size, num_workers=4)
        self.batch_size = batch_size
        self.max_len = max_len

    def get_predictions(self, data_loader: DataLoader):
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        with torch.no_grad():
            for d in tqdm(data_loader, desc="test"):
                texts = d["sent"]
                input_feats = d["feat"].to(self.device)
                targets = d["label"].float().to(self.device)
                preds = self.model(
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

    def test(self, test_loader: DataLoader) -> str:
        _, y_pred, y_pred_probs, y_test = self.get_predictions(test_loader)
        return classification_report(y_test, y_pred, target_names=["human", "machine"])



def main(ran: Optional[int] = 50):
    f = open(f'../data/final_results/logs_exp01.txt', mode="w")
    # general
    feat_model = "tree_small"
    tok_name = "prajjwal1/bert-tiny"
    para = AutoParaphraser("humarin/chatgpt_paraphraser_on_T5_base", cuda=True)

    # dep test
    m_dep = HumanMachineClassifierBertTiny()
    m_dep_path = "../data/feat_exp/best_model_state.bin"
    exp_dep = Experiment(model=m_dep,
                         model_path=m_dep_path,
                         feat_model=feat_model,
                         tok_name=tok_name
                         )
    paraphrased_testloader = exp_dep.paraphrase_ds(para, ran)
    exp_dep_report0 = exp_dep.test(exp_dep.test_data_loader)
    exp_dep_report1 = exp_dep.test(paraphrased_testloader)
    del exp_dep
    del m_dep

    print(100 * "=")
    print("BERT-FEATURES-BEFORE:")
    print(exp_dep_report0)
    print("\n")
    print("BERT-FEATURES-AFTER:")
    print(exp_dep_report1)
    print(100 * "=")

    f.write(100 * "=")
    f.write("\n")
    f.write("BERT-FEATURES-BEFORE:\n\n")
    f.write(exp_dep_report0)
    f.write("\n\n")
    f.write("BERT-FEATURES-AFTER:\n")
    f.write(exp_dep_report1)
    f.write("\n\n")
    f.write(100 * "=")
    f.write("\n\n")

    # dep test-feat
    m_dep_feat = HumanMachineClassifierBertTinyFeatureTiny()
    m_dep_feat_path = "../data/feat_dep_exp/best_model_state.bin"
    exp_dep_feat = Experiment(model=m_dep_feat,
                              model_path=m_dep_feat_path,
                              feat_model=feat_model,
                              tok_name=tok_name
                              )
    exp_dep_feat_report0 = exp_dep_feat.test(exp_dep_feat.test_data_loader)
    exp_dep_feat_report1 = exp_dep_feat.test(paraphrased_testloader)

    print(100 * "=")
    print("BERT+DEP-FEATURES-BEFORE:")
    print(exp_dep_feat_report0)
    print("\n")
    print("BERT+DEP-FEATURES-AFTER:")
    print(exp_dep_feat_report1)
    print(100 * "=")

    f.write(100 * "=")
    f.write("\n")
    f.write("BERT+DEP-FEATURES-BEFORE:\n\n")
    f.write(exp_dep_feat_report0)
    f.write("\n\n")
    f.write("BERT+DEP-FEATURES-AFTER:\n")
    f.write(exp_dep_feat_report1)
    f.write("\n\n")
    f.write(100 * "=")
    f.write("\n\n")

    f.flush()
    f.close()


def dep_form_exp(load: bool = False):
    # general
    paraphrased_dataset = Dataset.load_from_disk(f"../data/paraphrased_ds/para_set")
    if not load:
        dataset_after = dep_parse_spacy_dataset(paraphrased_dataset, save_path=f"../data/paraphrased_ds/para_set_depform")
    else:
        dataset_after = Dataset.load_from_disk(f"../data/paraphrased_ds/para_set_depform")
    dataset_before = DatasetDict.load_from_disk(f"../data/hm_dataset_prajjwal1-bert-tiny+tree_small+dep_tree")["test"]

    # dep test
    model = HumanMachineClassifierBertTiny()
    m_dep_path = "../data/dep_form_exp/best_model_state.bin"
    exp = Experiment2(model=model,
                      model_path=m_dep_path,
                      ds_before=dataset_before,
                      ds_after=dataset_after)

    exp_dep_report0 = exp.test(exp.test_data_loader_before)
    exp_dep_report1 = exp.test(exp.test_data_loader_after)
    print(100 * "=")
    print("BERT-FEATURES-BEFORE:")
    print(exp_dep_report0)
    print("\n")
    print("BERT-FEATURES-AFTER:")
    print(exp_dep_report1)
    print(100 * "=")


def feat_form_exp():
    # general
    dataset_after = Dataset.load_from_disk(f"../data/paraphrased_ds/para_set")

    dataset_before = DatasetDict.load_from_disk(f"../data/hm_dataset_prajjwal1-bert-tiny+tree_small")["test"]

    # dep test
    model = HumanMachineClassifierFeature5()
    m_dep_path = "../data/feat_exp/best_model_state.bin"
    exp = Experiment3(model=model,
                      model_path=m_dep_path,
                      ds_before=dataset_before,
                      ds_after=dataset_after)

    exp_dep_report0 = exp.test(exp.test_data_loader_before)
    exp_dep_report1 = exp.test(exp.test_data_loader_after)
    print(100 * "=")
    print("DEP-FEATURES-BEFORE:")
    print(exp_dep_report0)
    print("\n")
    print("DEP-FEATURES-AFTER:")
    print(exp_dep_report1)
    print(100 * "=")


if __name__ == "__main__":
    # main(None)
    # general
    """feat_model = "tree_small"
    tok_name = "prajjwal1/bert-tiny"
    para = AutoParaphraser("humarin/chatgpt_paraphraser_on_T5_base", cuda=True)

    # dep test
    m_dep = HumanMachineClassifierBertTiny()
    m_dep_path = "../data/feat_exp/best_model_state.bin"
    exp_dep = Experiment(model=m_dep,
                         model_path=m_dep_path,
                         feat_model=feat_model,
                         tok_name=tok_name
                         )
    paraphrased_testloader = exp_dep.paraphrase_ds(para, None, save_load="save")"""
    # dep_form_exp()
    feat_form_exp()
