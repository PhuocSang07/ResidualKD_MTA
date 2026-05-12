from torch.utils.data import Dataset
import torch
import json
from transformers import PreTrainedTokenizer

from dataclasses import dataclass


class LLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, prompt_max_len=512):
        self.dataset = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.dataset.append(data)

                s_prompt = tokenizer(
                    data['prompt'],
                    max_length=prompt_max_len,
                    truncation=True,
                    add_special_tokens=False
                )
                data['prompt'] = tokenizer.decode(s_prompt['input_ids'])
                data['prompt_len'] = len(s_prompt['input_ids'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (self.dataset[index]['prompt'],
                self.dataset[index]['output'],
                self.dataset[index]['prompt_len'])


@dataclass
class LLMDataCollator:
    tokenizer: PreTrainedTokenizer = None
    model_type: str = ''
    do_train: bool = True
    max_len: int = 512
    pad_to_multiple_of: int = 4
    return_tensors: str = 'pt'
    padding: bool = True

    def __call__(self, batch):
        prompts, fulls, prompt_lengths = [], [], []
        for prompt, output, prompt_length in batch:
            prompts.append(prompt)
            fulls.append(prompt + output)
            prompt_lengths.append(prompt_length)

        inputs = self.tokenizer(
            fulls,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len - 1,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=False
        )

        eos_tokens = torch.full((inputs["input_ids"].size(0), 1), self.tokenizer.eos_token_id, dtype=torch.long)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], eos_tokens], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"],
                                              torch.zeros((inputs["attention_mask"].size(0), 1), dtype=torch.long)], dim=1)

        labels = inputs["input_ids"][:, 1:].clone().detach()
        labels = torch.cat([labels, torch.full((labels.size(0), 1), -100, dtype=torch.long)], dim=1)

        input_lengths = inputs["attention_mask"].sum(dim=1)
        prompt_lengths = torch.tensor(prompt_lengths)

        if self.model_type in ["gpt2"]:
            position_ids = torch.zeros(inputs['input_ids'].size(), dtype=torch.long)
            for i in range(input_lengths.size(0)):
                position_ids[i, :input_lengths[i]] = torch.arange(0, input_lengths[i], dtype=torch.long)
            inputs["position_ids"] = position_ids

        for i in range(len(labels)):
            labels[i, :(prompt_lengths[i] - 1)] = -100
            labels[i, input_lengths[i]:] = -100

        if not self.do_train:
            return inputs, None, labels

        return inputs, labels
