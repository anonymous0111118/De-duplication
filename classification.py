from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, BertConfig, BertForPreTraining, BertTokenizer)
from tqdm import tqdm
from model import Model
import sys
ROOT_PATH = '/data/fanxingyu/deduplication/'
def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()
xianka = {'gcc430':'4', 'gcc440':'5', 'gcc450':'6', 'llvm280': '7'}

TYPE = 'gcc430'
os.environ['CUDA_VISIBLE_DEVICES'] = xianka[TYPE]
LOG_FILE = ROOT_PATH + 'logs/' + TYPE + '.log'
put_file_content(LOG_FILE, 'logging' + '\n')
MODEL_CLASSES = {
    'unixcoder': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codebert-cpp': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

bug = []

name2id = {}
id2name = {}

with open(ROOT_PATH + 'drawing/data/' + TYPE + '/names', 'r') as f:
    for index, line in enumerate(f):
        name = line.strip().split('/')[-1].split('.')[0]
        name2id[name] = index
        id2name[index] = name
miss = {}
miss['gcc430'] = []
miss['gcc440'] = ['417', '31', '65']
miss['gcc450'] = ['6', '11', '14','17','18', '24']
miss['llvm280'] = ['86', '100', '90','87']
xaxis = {'gcc430':29, 'gcc440':20, 'gcc450':7, 'llvm280':6}
miss_id = []
for i in miss[TYPE]:
    miss_id.append(name2id[i])

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args, idx):
    # print(1)
    # print(js)
    code = js.split('<CODESPLIT>')[0]
    # print(code)
    # sys.exit()
    try:
        target = int(js.split('<CODESPLIT>')[1].strip())
    except:
        target = 0
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(source_tokens, source_ids, idx, target)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        idx = 0
        with open(file_path) as f:
            for line in f:
                idx = idx + 1
                js = line
                convert_examples_to_features(js, tokenizer, args, idx)
                try:
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, idx))
                except:
                    print('???')
                    continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    print('training...')
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tr_loss = 0.0
    model.zero_grad()
    for idx in range(args.epochs):
        put_file_content(LOG_FILE, "epoch :  " + str(idx) + "\n")
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            if len(batch[0]) < args.train_batch_size:
                continue
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, _, _ = model(inputs, labels)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        results = evaluate(args, model, tokenizer, eval_when_training=True)
        print(results['eval_precision'])
 

def evaluate(args, model, tokenizer, eval_when_training=False):
    print('evaluating...')
    # build dataloader

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    all_list = []
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit, inputs_lastlayer = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    
    # logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = []
    for logit in logits:
        for log in logit:
            y_preds.append(np.argmax(log))
    y_preds = np.array(y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }
    checkpoint_prefix = 'saved_models' + TYPE
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_dir = os.path.join(output_dir,
                                '{}'.format('model.bin'))
    torch.save(model_to_save.state_dict(), output_dir)
    print("Saving model checkpoint to %s", output_dir)
    return result


def main():
    with open(ROOT_PATH  + '/datasets/' + TYPE + '/bugs.txt', 'r') as f:
        for eachLine in f:
            bug.append(eachLine.strip())
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=0, type=int)
    parser.add_argument("--eval_data_file",
                        default= ROOT_PATH + '/datasets/' + TYPE + '/' + TYPE + '-oversamp.txt',
                        type=str,
                        help="eval data file")
    parser.add_argument("--train_data_file",

                        default= ROOT_PATH + '/datasets/' + TYPE + '/' + TYPE + '-oversamp.txt',
                        type=str,
                        help="train data file")
    args = parser.parse_args()
    args.output_dir = './saved_models_man/' + TYPE + '/'
    # 修改
    args.number_labels = 1

    args.block_size =  256
    args.seed = 123456
    args.evaluate_during_training = True
    # 修改
    args.language_type = 'c'
    args.train_batch_size = 64
    args.eval_batch_size = 64
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.max_steps = -1
    args.adam_epsilon = 1e-8
    args.weight_decay = 0.0
    args.gradient_accumulation_steps = 1
    pretrain = str(ROOT_PATH+'pretrained_model/unixcoder-base')
    args.model_type = 'unixcoder' 
    args.config_name = pretrain
    args.model_name_or_path = pretrain
    args.tokenizer_name = pretrain
    args.epochs = 500
    args.learning_rate = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(pretrain)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    

    model = Model(model, config, tokenizer, args)
    model.to(args.device)
    

    train_dataset = TextDataset(tokenizer, args, args.train_data_file)


    print('do train')
    train(args, train_dataset, model, tokenizer)


if __name__ == "__main__":
    main()



'''
bug158782 number:674
bug133004 number:189
bug156496 number:144
bug139094 number:101
'''
