import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
bug = []
TYPE = 'gcc430'
with open(TYPE+'-bugs.txt', 'r') as f:
    for eachLine in f:
        bug.append(eachLine.strip())
# 576.5
# 450 : 1 4 14 10 24 6

name2id = {}
id2name = {}

with open('data/'+TYPE+'/names', 'r') as f:
    for index, line in enumerate(f):
        name = line.strip().split('/')[-1].split('.')[0]
        name2id[name] = index
        id2name[index] = name
miss = {}
miss['gcc430'] = []
miss['gcc440'] = ['417', '31', '65']
miss['gcc450'] = ['6', '11', '14','17','18', '24']
miss['llvm280'] = ['86', '100', '90','87']


miss_id = []
for i in miss[TYPE]:
    miss_id.append(name2id[i])
for i in miss_id:
    bug[i] = bug[1]
def fpf(disname, start): 
    ll = np.load(disname)
    res = [] # result
    dis = [] # the distance to main set
    vis = set() # has been visited
    bugset = set() # the size of bug found
    rank = []
    score = 0
    lenll = len(ll)
    vis.add(start)
    for i in miss_id:
        vis.add(i)
    area = 0.0
    bugset.add(bug[start])
    rank.append(start)
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
        rank.append(maxxnum)
        res.append(len(bugset))
        area += len(bugset)
        for j in range(lenll):
            dis[j] = min(dis[j], ll[maxxnum][j])

    return rank, res.copy(), area, np.array(resx), ll


def draw(dis, labels, topn):
    topn = min(topn, len(dis[0]))
    bug_num = int(dis[0][-1])
    theoretical_best = np.zeros(3000)
    if topn < bug_num:
        theoretical_best = list(range(1, topn + 1))
    else:
        theoretical_best = list(range(1, bug_num + 1))
        for i in range(bug_num + 1, topn + 1):
            theoretical_best.append(bug_num)
    best_auc = np.trapz(theoretical_best)
    theoretical_best = np.array(theoretical_best)     
    raucs = []
    for num, t in zip(dis, labels):
        num = num[:topn]
        auc = np.trapz(num)
        rauc = auc / best_auc
        rauc = str(rauc * 100)[0:5] + '%'
        raucs.append(rauc)
        print(raucs)
        num = np.insert(num, 0, 0)
        line , = plt.plot(num, label=t + ' ' + '(RAUC: ' + rauc + ')', lw=2.8)
        plt.legend()
    plt.savefig(f'/data/fanxingyu/deduplication/drawing/save_pictures/' + TYPE + '.jpg')
    plt.close()


if __name__ == '__main__':
    res = [] 
    area = 0.0
    best_score = -1
    start = -1
    area = 0
    times = 0.0
    x = np.full((5,len(set(bug))), 0.0)
    for epoch in tqdm(range(20)):
        times += 1
        ans = []
        a, b, c, d, e = fpf('distances/' + TYPE + '/'+TYPE+'-w.npy', epoch)
        ans.append(b)
        x[0] += d
        a, b, c, d, e = fpf('distances/' + TYPE + '/tamer.npy', epoch)
        ans.append(b)
        x[1] += d
        a, b, c, d, e = fpf('distances/' + TYPE + '/transformer.npy', epoch)
        x[2] += d
        ans.append(b)
        a, b, c, d, e = fpf('distances/' + TYPE + '/newD3.npy', epoch)
        x[3] += d
        ans.append(b)
        a, b, c, d, e = fpf('distances/' + TYPE + '/programnew.npy', epoch)
        x[4] += d
        ans.append(b)
        res.append(ans.copy())# 20 * 4 * n
    for i in range(len(x)):
        x[i] /= times

    outstr = ''
    mm = {'gcc430':29, 'gcc440':20, 'gcc450':7, 'llvm280':6}

    ll = np.array(res)
    ll = np.transpose(ll, (1, 2, 0))
    ll = np.mean(ll, axis = 2)
    ll = ll.tolist()
    labels = [ 'BLADE','tamer',  'trans', 'D3', 'D3-prog']
    
    raucs = draw(ll, labels, int(len(ll[0])))
