'''
This part is used to train the dialect model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from utils.tools import *
from utils.loss import AAMsoftmax, StandardSimilarityLoss, PairDistanceLoss
from utils.model import ECAPA_TDNN
import pandas as pd

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, loss, device, **kwargs):
        super(ECAPAModel, self).__init__()
        ## model
        self.device = device
        self.dialect_encoder = ECAPA_TDNN(C = C).to(self.device)
         
        ##loss    
        if loss == 'aamsoftmax':
            self.dialect_loss = AAMsoftmax(n_class = n_class, m = m, s = s).to(self.device)
        elif loss == 'StandardSimilarityLoss':
            self.dialect_loss = StandardSimilarityLoss(n_class).to(self.device)
        elif loss == 'PairDistanceLoss':
            self.dialect_loss = PairDistanceLoss().to(self.device)
        else:
            raise NotImplementedError
        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.dialect_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, loss = 0, 0
        lr = self.optim.param_groups[0]['lr']
        
        for num, (data, labels) in enumerate(loader, start = 1):
            self.zero_grad()
            labels = torch.LongTensor(labels).to(self.device)
            dialect_embedding = self.dialect_encoder.forward(data.to(self.device), aug = True)
            nloss = self.dialect_loss.forward(dialect_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "%(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f\r"%(loss/(num)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr

    def eval_network(self, loader):
        self.eval()

        total_loss = 0.0
        for idx, (data, label) in tqdm.tqdm(enumerate(loader)):
            data = data.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                embedding = self.dialect_encoder.forward(data, aug = False)
                loss = self.dialect_loss(embedding, label)
                total_loss += loss.item()
        total_loss = total_loss / (idx + 1)
        return total_loss

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)