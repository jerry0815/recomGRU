import os
import time
import torch
import argparse
from tqdm import tqdm
from model import recomGRU
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=10, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=81, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = recomGRU(itemnum, 50, 2, args.maxlen,args.hidden_units).to(args.device) # no ReLU activation in original SASRec implementation?
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset)
        print('test (NDCG@20: %.4f, HR@20: %.4f) (NDCG@10: %.4f, HR@10: %.4f) (NDCG@5: %.4f, HR@5: %.4f)' % \
            (t_test[0], t_test[1],t_test[2], t_test[3],t_test[4], t_test[5]))
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    # epoch_start_idx = 1
    # for hidden in range(20,50,10):
    #     print("Start with hidden: " ,hidden)
    #     args.hidden_units = hidden
    #     f = open(os.path.join(args.dataset + '_' + args.train_dir + 'log_' + str(hidden) + '.txt'), 'w')
    #     sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    #     model = recomGRU(itemnum, 50, 2, args.maxlen,args.hidden_units).to(args.device) # no ReLU activation in original SASRec implementation?
    #     criterion = torch.nn.BCEWithLogitsLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    T = 0.0
    t0 = time.time()

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        train_loss = 0.0
        train(model,sampler,optimizer, criterion, args.device, num_batch)

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset)
            t_valid = evaluate_valid(model, dataset)
            print('valid (NDCG@20: %.4f, HR@20: %.4f) (NDCG@10: %.4f, HR@10: %.4f) (NDCG@5: %.4f, HR@5: %.4f)' % \
            (t_valid[0], t_valid[1],t_valid[2], t_valid[3],t_valid[4], t_valid[5]))
            print('test (NDCG@20: %.4f, HR@20: %.4f) (NDCG@10: %.4f, HR@10: %.4f) (NDCG@5: %.4f, HR@5: %.4f)' % \
            (t_test[0], t_test[1],t_test[2], t_test[3],t_test[4], t_test[5]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            #f.flush()
            t0 = time.time()
            model.train()


        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'recomGRU.epoch={}.lr={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    f.close()
    sampler.close()
    print("Done")