from torch import nn, optim
from transformers import BertModel, BertTokenizer


class HumanMachineClassifierBert(nn.Module):
    def __init__(self, tok_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(tok_name)
        self.tokenizer = BertTokenizer.from_pretrained(tok_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


class Trainer