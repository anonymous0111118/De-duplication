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
import matplotlib.pyplot as plt
import sys
ROOT_PATH = '/data/fanxingyu/deduplication/'
def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
TYPE = 'gcc430'
LOG_FILE = ROOT_PATH + 'logs/' + TYPE + '.log'
put_file_content(LOG_FILE, 'logging' + '\n')
# 修改
MODEL_CLASSES = {
    'unixcoder': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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
xaxis = {'gcc430':29}
miss_id = []
for i in miss[TYPE]:
    miss_id.append(name2id[i])

def fpf(ll, start): 
    res = [] # result
    dis = [] # the distance to main set
    vis = set() # has been visited
    bugset = set() # the size of bug found
    score = 0
    lenll = len(ll)
    vis.add(start)
    for i in miss_id:
        vis.add(i)
    area = 0.0
    bugset.add(bug[start])
    # print(bug[start])
    bugs = []
    resx = []
    bugs.append(bug[start])
    res.append(1)
    resx.append(1)
    area += 1
    for i in range(lenll):
        dis.append(ll[start][i])
    lenvis = len(vis)
    for i in range(lenll-lenvis):
        maxxnum = 1e9
        maxxdis = -1e9
        for j in range(lenll):
            if j not in vis and dis[j] > maxxdis:
                maxxnum = j 
                maxxdis = dis[j]

        vis.add(maxxnum)
        if bug[maxxnum] not in bugset:
            bugs.append(bug[maxxnum])
            resx.append(i + 2)
        bugset.add(bug[maxxnum])
        res.append(len(bugset))
        if i < 100:
            score += maxxdis
        area += len(bugset)
        for j in range(lenll):
            dis[j] = min(dis[j], ll[maxxnum][j])
    apfd = APFD(res)
    return apfd, res.copy(), area, np.array(resx)

def APFD(res):
    n = len(res)
    if TYPE == 'gcc430':
        m = 29
    elif TYPE == 'gcc440':
        m = 20
    elif TYPE == 'gcc450':
        m = 7
    else:
        m = 6
    tf = 1.0
    print(m)
    for i in range(len(res)):
        if i and res[i] > res[i-1]:
            # print(i)
            tf += i + 1 
    ans = 1.0 - tf/float(n*m) + 1.0/2/n
  
    return ans
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


def get_distance(args, model, tokenizer, eval_when_training=False):
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
            for i in inputs_lastlayer.cpu().numpy():
                all_list.append(i)
        nb_eval_steps += 1
    w = list(model.out_proj.parameters())[0].data.cpu().detach().numpy()[0]
    w = np.where(w < 0, -w, w)
    ll = []
    for i in tqdm(range(len(all_list))):
        l1 = []
        for j in range(len(all_list)):
            # l1.append(np.sqrt(np.sum(np.square(w*all_list[i] - w*all_list[j]))))
            l1.append(np.sum( w * np.abs(all_list[i] - all_list[j]) ))
            # mid = np.dot(  w * all_list[i],   w*all_list[j])/ ( np.linalg.norm( w*all_list[i]) * np.linalg.norm( w*all_list[j]) )
            # l1.append((2 - mid) / 2.0)
        ll.append(l1.copy())
    ll = np.array(ll)
    np.save(TYPE + 'BLADE.npy', ll, allow_pickle=True, fix_imports= True)


def main():
    with open(ROOT_PATH  + '/datasets/' + TYPE + '/bugs.txt', 'r') as f:
        for eachLine in f:
            bug.append(eachLine.strip())
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=0, type=int)
    parser.add_argument("--eval_data_file",
                        default= ROOT_PATH + '/datasets/' + TYPE + '/' + TYPE + '-code-fail.txt',
                        type=str,
                        help="eval data file")
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
    args.eval_batch_size = 8
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
    args.epochs = 3000
    args.learning_rate = 5e-3
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
    model.load_state_dict(torch.load('saved_models' + TYPE + '/model.bin'))
    get_distance(args, model, tokenizer)


if __name__ == "__main__":
    main()



'''
bug158782 number:674
bug133004 number:189
bug156496 number:144
bug139094 number:101
'''
