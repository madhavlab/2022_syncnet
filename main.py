import torch
import json
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import norm
from tqdm.auto import tqdm
import IPython
import os
import glob
from syncnet import SyncNet, hybrid_loss
from utils import prepare_data
from utils import supplementary
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SyncNet(gamma=0.125,num_towers=50,causal_tower_height=5,tower_filters=[0,0,0,0,0,54,50,32,16,16]).to(device)
data = prepare_data(m=model,overwrite=False,train=0.85)
dataloader = torch.utils.data.DataLoader(data,batch_size=16,num_workers=1,pin_memory=True,shuffle=False)
torch.backends.cudnn.benchmark = True
supp=supplementary()

current_epoch=0
path = os.path.join(os.getcwd(),'checkpoints')
try:
  os.makedirs(os.path.join(path,'models'))
  os.makedirs(os.path.join(path,'loss'))
except:
  pass
def train(model,train_dl,num_epochs,checkpoint=None):
  criterion = hybrid_loss(l1=0.3,l2=26,l3=4500)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0014,
                                                  steps_per_epoch=int(len(train_dl)),
                                                  epochs=num_epochs,
                                                  anneal_strategy='linear')
  running_loss = {}
  if checkpoint:
    print('loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    scheduler.load_state_dict(checkpoint['scheduler'])
    model.train()

  for epoch in tqdm(range(num_epochs)):
    running_loss['epoch'+str(current_epoch+epoch+1)]=[]
    for i,sample in enumerate(train_dl):
      input_seq, target_corr_seq, target_pool_seq, weights_corr, weights_pool = sample['input_sequence'].to(device).float(),\
       sample['target_sequence_corr'].to(device).float(),\
       sample['target_sequence_pooled'].to(device).float(),\
       sample['weights_corr'].to(device),\
       sample['weights_pool'].to(device)
      optimizer.zero_grad()
      predicted_corr_seq = model(input_seq)
      predicted_corr_seq = supp.correlation(predicted_corr_seq)
      loss = criterion(predicted_corr_seq,supp.normalized(target_corr_seq),target_pool_seq,weights_corr,weights_pool)
      running_loss['epoch'+str(current_epoch+epoch+1)].append(loss.mean().item())
      loss.backward()
      optimizer.step()
    scheduler.step()
    if (epoch+1)%3==0:
      f = os.path.join(path+'/models', 'checkpoint.pth')
      torch.save({'epoch': current_epoch+epoch+1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss':loss,
                  'running_loss': running_loss,
                  'scheduler':scheduler.state_dict()}, f)
      print('epoch {} done. Loss is {}'.format(current_epoch+epoch+1,np.mean(running_loss['epoch'+str(current_epoch+epoch+1)])))
  return running_loss

model = SyncNet(gamma=0.125,num_towers=50,causal_tower_height=5,tower_filters=[0,0,0,0,0,54,50,32,16,16]).to(device)
if current_epoch !=0:
  print('loaded!')
  checkpoint = torch.load(path+'/models/checkpoint.pth'.format(current_epoch))
else:
  checkpoint=None
running_loss = train(model=model,train_dl=dataloader,num_epochs=150,checkpoint=checkpoint)