import numpy as np
import torch
import torch.nn as nn

class recomGRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_layers, max_len, hidden_size):
        super(recomGRU, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.look_up = nn.Embedding(input_dim+1, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size, n_layers, dropout = 0.5)
        self.linear1 = nn.Linear(self.hidden_size, embedding_dim)
    
        self.activate = nn.Softmax()

    def forward(self,input, hidden, pos, neg):
        #embedded = input.unsqueeze(0)
        input = self.look_up(torch.LongTensor(input).to(self.device))
        pos = self.look_up(torch.LongTensor(pos).to(self.device))
        neg = self.look_up(torch.LongTensor(neg).to(self.device))
        output,hidden = self.gru(input, hidden)  

        # print(output.shape)
        # print(pos.shape)
        logit = self.linear1(output)
        pos_logit = (logit * pos).sum(dim=-1)
        neg_logit = (logit * neg).sum(dim=-1)
        #print(pos_logit.shape)
        return pos_logit, neg_logit, hidden 

    def predict(self,input, hidden, item):
        #embedded = input.unsqueeze(0)
        input = self.look_up(torch.LongTensor(input).to(self.device))
        item = self.look_up(torch.LongTensor(item).to(self.device))

        #print("predict: ", hidden.shape)
        output,hidden = self.gru(input, hidden)  
        logit = self.linear1(output)
        logit = logit[:,-1,:]

        logits = item.matmul(logit.unsqueeze(-1)).squeeze(-1)

        return logits, hidden

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.n_layers, self.max_len, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.n_layers, self.max_len, self.hidden_size).to(self.device)
        return h0