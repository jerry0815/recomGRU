import sys
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def train(model, sampler, optimizer, criterion, device, num_batch):

    train_loss = 0.0
    model.train()     # Enter Train Mode
    hidden = model.init_hidden()
    for step in range(num_batch):#, total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

        hidden = hidden.detach()
        pos_logits, neg_logits, hidden = model(seq, hidden, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        optimizer.zero_grad()# adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = criterion(pos_logits[indices], pos_labels[indices])#bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        loss.backward()
        optimizer.step()#adam_optimizer.step()
        # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        train_loss += loss.item()

    return train_loss


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset):

    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    maxlen = 20
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    NDCG20 = 0.0
    HT20 = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    with torch.no_grad():
        hidden = model.init_hidden()
        for u in users:
            if len(train[u]) < 1 or len(test[u]) < 1: continue
            seq = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            #101
            # rated = set(train[u])
            # rated.add(0)

            # item_idx = [test[u][0]]
            # for _ in range(100):
            #     t = np.random.randint(1, itemnum + 1)
            #     while t in rated: t = np.random.randint(1, itemnum + 1)
            #     item_idx.append(t)
            
            #All
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            rated.add(test[u][0])
            item_idx = item_idx + list(set(range(1,itemnum+1)) - rated)

            
            predictions, hidden = model.predict(np.array([seq]), hidden, np.array(item_idx))
            predictions = -predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1
                if rank < 10:
                    if rank < 5:
                        NDCG5 += 1 / np.log2(rank + 2)
                        HT5 += 1
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG20 / valid_user, HT20 / valid_user, NDCG / valid_user, HT / valid_user, NDCG5 / valid_user, HT5 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset):
    model.eval()

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    maxlen = 20
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    NDCG20 = 0.0
    HT20 = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    with torch.no_grad():
        hidden = model.init_hidden()
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            hidden = hidden.detach()
            predictions, hidden = model.predict(np.array([seq]), hidden, np.array(item_idx))
            predictions = -predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1
                if rank < 10:
                    if rank < 5:
                        NDCG5 += 1 / np.log2(rank + 2)
                        HT5 += 1
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG20 / valid_user, HT20 / valid_user, NDCG / valid_user, HT / valid_user, NDCG5 / valid_user, HT5 / valid_user