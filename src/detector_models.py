from collections import defaultdict
from typing import Optional

from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from extract_dataset import load_dataset_info, get_dataloaders, construct_dataset
from textscorer import ScorerModel


class HumanMachineClassifierBert(nn.Module):
    def __init__(self, tok_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(tok_name)
        self.tokenizer = BertTokenizer.from_pretrained(tok_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask).pooler_output
        output = self.drop(pooled_output)
        output = self.out(output)
        # return self.softmax(output)
        return self.sigmoid(output)


class BertFeatureTrainer:
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 8,
                 epochs: int = 10,
                 device: str = "cuda:0",
                 max_steps: Optional[int] = None,
                 lr_rate: float = 2e-5,
                 warm_up_steps: int = 2000
                 ):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=lr_rate, correct_bias=False)
        dataset = load_dataset_info(do_print=False)
        if max_steps is not None:
            max_steps = max_steps // epochs
            dataset["train"] = dataset["train"].select(range(int(max_steps * 0.8)))
            dataset["test"] = dataset["test"].select(range(int(max_steps * 0.1)))
            dataset["validation"] = dataset["validation"].select(range(int(max_steps * 0.1)))

        print("DATASET:")
        print(f'train-size: {len(dataset["train"])}')
        print(f'test-size: {len(dataset["test"])}')
        print(f'val-size: {len(dataset["validation"])}')

        self.n_samples_train = len(dataset["train"])
        self.n_samples_test = len(dataset["test"])
        self.n_samples_val = len(dataset["validation"])
        trl, tel, val = get_dataloaders(dataset, batch_size=batch_size)
        self.train_data_loader = trl
        self.test_data_loader = tel
        self.val_data_loader = val
        self.total_steps = len(self.train_data_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warm_up_steps,
                                                         num_training_steps=self.total_steps)
        # self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.loss_fn = nn.BCELoss().to(self.device)

    def train_epoch(self):
        model = self.model.train()
        losses = []
        correct_predictions = 0
        for d in tqdm(self.train_data_loader, desc="train"):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["label"].float().to(self.device)
            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).flatten()
            # _, preds = torch.max(outputs, dim=1)
            #print(preds)
            #print(targets)
            loss = self.loss_fn(preds, targets)
            correct_predictions += torch.sum(preds.round() == targets)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
        return correct_predictions.double() / self.n_samples_train, np.mean(losses)

    def eval_model(self):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in tqdm(self.val_data_loader, desc="eval"):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label"].float().to(self.device)
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(preds, targets)
                correct_predictions += torch.sum(preds.round() == targets)
                losses.append(loss.item())
        return correct_predictions.double() / self.n_samples_val, np.mean(losses)

    def test_model(self):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in self.test_data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label"].float().to(self.device)
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(preds, targets)
                correct_predictions += torch.sum(preds.round() == targets)
                losses.append(loss.item())
        return correct_predictions.double() / self.n_samples_test, np.mean(losses)

    def get_predictions(self):
        model = self.model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        with torch.no_grad():
            for d in tqdm(self.test_data_loader, desc="test"):
                texts = d["sent"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label"].float().to(self.device)
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                review_texts.extend(texts)
                predictions.extend(preds.round())
                prediction_probs.extend(preds)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        print(predictions)
        print(prediction_probs)
        print(real_values)
        return review_texts, predictions, prediction_probs, real_values

    def train(self):
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch()
            print(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = self.eval_model()
            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), '../data/best_model_state.bin')
                best_accuracy = val_acc

            if epoch % 3 == 0:
                # test_acc, _ = self.test_model()
                _, y_pred, y_pred_probs, y_test = self.get_predictions()
                print(f"EPOCH: {epoch}")
                # print(test_acc)
                print(classification_report(y_test, y_pred, target_names=["human", "machine"]))


class HumanMachineClassifierFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(40, 40)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(40, 40)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(40, 40)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(40, 1)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, sent):
        x = self.act1(self.layer1(sent))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


class FeatureTrainer:
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 64,
                 epochs: int = 15,
                 device: str = "cpu",
                 max_steps: Optional[int] = None,
                 lr_rate: float = 2e-5,
                 warm_up_steps: int = 2000
                 ):
        self.device = device
        self.model = model
        self.scorer = ScorerModel()
        self.model.to(self.device)
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=lr_rate, correct_bias=False)
        dataset = load_dataset_info(do_print=False, format_torch=False)
        if max_steps is not None:
            max_steps = max_steps // epochs
            dataset["train"] = dataset["train"].select(range(int(max_steps * 0.8)))
            dataset["test"] = dataset["test"].select(range(int(max_steps * 0.1)))
            dataset["validation"] = dataset["validation"].select(range(int(max_steps * 0.1)))

        dataset["train"] = dataset["train"].remove_columns(["input_ids", "attention_mask"])
        dataset["test"] = dataset["test"].remove_columns(["input_ids", "attention_mask"])
        dataset["validation"] = dataset["validation"].remove_columns(["input_ids", "attention_mask"])
        print("DATASET:")
        print(f'train-size: {len(dataset["train"])}')
        print(f'test-size: {len(dataset["test"])}')
        print(f'val-size: {len(dataset["validation"])}')

        self.n_samples_train = len(dataset["train"])
        self.n_samples_test = len(dataset["test"])
        self.n_samples_val = len(dataset["validation"])
        trl, tel, val = get_dataloaders(dataset, batch_size=batch_size)
        self.train_data_loader = trl
        self.test_data_loader = tel
        self.val_data_loader = val
        self.total_steps = len(self.train_data_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warm_up_steps,
                                                         num_training_steps=self.total_steps)
        # self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.loss_fn = nn.BCELoss().to(self.device)

    def train_epoch(self):
        model = self.model.train()
        losses = []
        correct_predictions = 0
        for d in tqdm(self.train_data_loader, desc="train"):
            targets = d["label"].float().to(self.device)
            input = self.scorer(d["sent"]).to(self.device)
            preds = model(
                sent=input
            ).flatten()
            # _, preds = torch.max(outputs, dim=1)
            #print(preds)
            #print(targets)
            loss = self.loss_fn(preds, targets)
            correct_predictions += torch.sum(preds.round() == targets)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
        return correct_predictions.double() / self.n_samples_train, np.mean(losses)

    def eval_model(self):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in tqdm(self.val_data_loader, desc="eval"):
                targets = d["label"].float().to(self.device)
                input = self.scorer(d["sent"]).to(self.device)
                preds = model(
                    sent=input
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(preds, targets)
                correct_predictions += torch.sum(preds.round() == targets)
                losses.append(loss.item())
        return correct_predictions.double() / self.n_samples_val, np.mean(losses)

    def test_model(self):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in self.test_data_loader:
                targets = d["label"].float().to(self.device)
                input = self.scorer(d["sent"]).to(self.device)
                preds = model(
                    sent=input
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(preds, targets)
                correct_predictions += torch.sum(preds.round() == targets)
                losses.append(loss.item())
        return correct_predictions.double() / self.n_samples_test, np.mean(losses)

    def get_predictions(self):
        model = self.model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        with torch.no_grad():
            for d in tqdm(self.test_data_loader, desc="test"):
                texts = d["sent"]
                targets = d["label"].float().to(self.device)
                input = self.scorer(d["sent"]).to(self.device)
                preds = model(
                    sent=input
                ).flatten()
                # _, preds = torch.max(outputs, dim=1)
                review_texts.extend(texts)
                predictions.extend(preds.round())
                prediction_probs.extend(preds)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        print(predictions)
        print(prediction_probs)
        print(real_values)
        return review_texts, predictions, prediction_probs, real_values

    def train(self):
        history = defaultdict(list)
        best_accuracy = 0
        progress = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch()
            print(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = self.eval_model()
            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), '../data/best_feature_model_state.bin')
                best_accuracy = val_acc

            if val_acc >= history['val_acc'][-2]:
                progress = 0
            else:
                progress += 1

            if progress > 5:
                break

            if epoch % 3 == 0:
                # test_acc, _ = self.test_model()
                _, y_pred, y_pred_probs, y_test = self.get_predictions()
                print(f"EPOCH: {epoch}")
                # print(test_acc)
                print(classification_report(y_test, y_pred, target_names=["human", "machine"]))


if __name__ == "__main__":
    tok_name = "prajjwal1/bert-tiny"
    # construct_dataset(tok_name=tok_name)
    model = HumanMachineClassifierBert(tok_name=tok_name)
    trainer = BertFeatureTrainer(model, batch_size=64, epochs=100).train()
    # trainer = Trainer(model, batch_size=4, max_steps=10000, lr_rate=5e-6).train()
    # trainer = Trainer(model, batch_size=16, device="cpu").train()

    """model = HumanMachineClassifierFeature()
    trainer = FeatureTrainer(model, batch_size=32, epochs=15, device="cuda:0").train()"""


