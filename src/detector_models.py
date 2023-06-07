from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from extract_dataset import load_dataset_info, get_dataloaders


class HumanMachineClassifierBert(nn.Module):
    def __init__(self, tok_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(tok_name)
        self.tokenizer = BertTokenizer.from_pretrained(tok_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask).pooler_output
        output = self.drop(pooled_output)
        output = self.out(output)
        # return self.softmax(output)
        return output


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 8,
                 epochs: int = 10,
                 device: str = "cuda:0"
                 ):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        dataset = load_dataset_info()
        self.n_samples_train = len(dataset["train"])
        self.n_samples_test = len(dataset["test"])
        self.n_samples_val = len(dataset["validation"])
        trl, tel, val = get_dataloaders(dataset, batch_size=batch_size)
        self.train_data_loader = trl
        self.test_data_loader = tel
        self.val_data_loader = val
        self.total_steps = len(self.train_data_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train_epoch(self):
        model = self.model.train()
        losses = []
        correct_predictions = 0
        for d in tqdm(self.train_data_loader, desc="train"):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["label"].to(self.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return correct_predictions.double() / self.n_samples_train, np.mean(losses)

    def eval_model(self):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in tqdm(self.val_data_loader, desc="eval"):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label"].to(self.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
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
                targets = d["label"].to(self.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
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
                targets = d["label"].to(self.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(outputs)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
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


if __name__ == "__main__":
    model = HumanMachineClassifierBert()
    # trainer = Trainer(model, batch_size=4).train()
    trainer = Trainer(model, batch_size=16, device="cpu").train()

