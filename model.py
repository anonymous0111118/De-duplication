# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import sys

class UnixcoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = self.dense1(features)
        x = F.tanh(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = UnixcoderClassificationHead(config)
        self.args = args
        self.query = 0
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)


    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = outputs['hidden_states'][-1]
        sep_mask = input_ids.eq(self.tokenizer.sep_token_id)
        if len(torch.unique(sep_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <sep> tokens.")

        outputs = hidden_states[sep_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]


        logits = self.dense1(outputs)
        last_input = F.tanh(logits)
        logits = self.out_proj(last_input)

        prob = F.softmax(logits, dim=1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, last_input
        else:
            return prob

