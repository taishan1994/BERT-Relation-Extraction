import os
import json
import torch
import numpy as np

from config import ReConfig
from model import BertRe
from data_loader import ReDataset, ReCollate

from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 dev_loader=None,
                 test_loader=None,
                 save_step=500,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                token_type_ids = batch_data["token_type_ids"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, token_type_ids, labels)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                global_step += 1

                print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()}")
                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_re.bin"))
                
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_re.bin"))

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_re.bin")))
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            token_type_ids = batch_data["token_type_ids"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask, token_type_ids, labels)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, -1)
            labels = labels.detach().cpu().numpy()

            preds.extend(logits)
            trues.extend(labels)

        report = classification_report(trues, preds, target_names=self.id2label.values())
        return report


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(data_name):
    args = ReConfig(data_name)

    with open(os.path.join(args.output_dir, "re_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, "train.txt"), "r") as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev.txt"), "r") as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    re_collate = ReCollate(args, tokenizer)
    train_dataset = ReDataset(train_data)
    dev_dataset = ReDataset(dev_data)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2,
                              collate_fn=re_collate.collate)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2,
                            collate_fn=re_collate.collate)

    model = BertRe(args)

    # for name,_ in model.named_parameters():
    #   print(name)

    model.to(device)
    t_toal = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label
    )

    train.train()

    report = train.test()
    print(report)


if __name__ == "__main__":
    data_name = "dgre"
    main(data_name)
