import torch
import numpy as np

from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data


class ReDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ReCollate:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def collate(self, batch_data):
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        for d in batch_data:
            text = d["text"]
            labels = d["labels"]
            h = labels[0]
            t = labels[1]
            label = labels[2]
            # [CLS]h[SEP]t[SEP]text[SEP]
            pre_length = 4 + len(h) + len(t)
            if len(text) > self.max_seq_len - pre_length:
                text = text[:self.max_seq_len - pre_length]
            if h not in text or t not in text:
                continue
            tmp_input_ids = self.tokenizer.tokenize("[CLS]" + h + "[SEP]" + t + "[SEP]" + text + "[SEP]")
            tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_input_ids)
            attention_mask = [1] * len(tmp_input_ids)
            input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
            attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
            token_type_ids = [0] * self.max_seq_len
            label = self.label2id[label]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(label)

        input_ids = torch.tensor(np.array(batch_input_ids))
        attention_mask = torch.tensor(np.array(batch_attention_mask))
        token_type_ids = torch.tensor(np.array(batch_token_type_ids))
        labels = torch.tensor(np.array(batch_labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return data
