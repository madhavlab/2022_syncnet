import torch
import torch.nn as nn
import torchaudio
import numpy as np
import torchvision
import torch.nn.functional as F
from utils import supplementary


supp = supplementary()
class CausalTower(nn.Module):
  def __init__(self,tower_filters):
    super(CausalTower, self).__init__()
    self.signal = torch.Tensor()
    #for i,layer_filter in enumerate(tower_filters):
    #  if i ==len(tower_filters)-1:
    #    self.tower.append(nn.Conv1d(in_channels=2,out_channels=2,kernel_size=int(layer_filter),bias=False))
    #  else:
    #    self.tower.append(nn.Conv1d(in_channels=2,out_channels=2,kernel_size=int(layer_filter)))
    #self.tower.append(nn.BatchNorm1d(num_features=2))
    # self.tower = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=4,kernel_size=60,dilation=3),
    #                             nn.Sequential(*[nn.Conv1d(in_channels=4,out_channels=6,kernel_size=96,bias=False,dilation=4),
    #                                             nn.BatchNorm1d(num_features=6)]),
    #                             nn.Conv1d(in_channels=6,out_channels=8,kernel_size=54,dilation=4),
    #                             nn.Sequential(*[nn.Conv1d(in_channels=8,out_channels=10,kernel_size=54,bias=False,dilation=4),
    #                                             nn.BatchNorm1d(num_features=10)]),
    #                             nn.Conv1d(in_channels=10,out_channels=12,kernel_size=54,dilation=4),
    #                             nn.Sequential(*[nn.Conv1d(in_channels=12,out_channels=14,kernel_size=54,bias=False,dilation=4),
    #                                             nn.BatchNorm1d(num_features=14)])])
    
    self.tower = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=4,kernel_size=60,dilation=6),
                                nn.Sequential(*[nn.Conv1d(in_channels=4,out_channels=8,kernel_size=72,bias=False,dilation=6),
                                                nn.BatchNorm1d(num_features=8)]),
                                nn.Conv1d(in_channels=8,out_channels=8,kernel_size=54,dilation=6),
                                nn.Sequential(*[nn.Conv1d(in_channels=8,out_channels=10,kernel_size=54,bias=False,dilation=6),
                                                nn.BatchNorm1d(num_features=10)]),
                                nn.Conv1d(in_channels=10,out_channels=12,kernel_size=54,dilation=5)])
        
  def forward(self,window,prev_tower_outputs=None):
    y = self.signal[:,:,window[0]:window[1]]
    result={}
    for id,layer in enumerate(self.tower):
      try:
        y.shape[2]==prev_tower_outputs['layer'+str(id)].shape[2]
        input = y+prev_tower_outputs['layer'+str(id)]
      except:
        input = y 
      y = torch.relu(layer(input))
      result['layer'+str(id+1)] = y
    result['layer0'] = 0
    return result


class SyncNet(nn.Module):

  def __init__(self,num_towers=50,tower_filters=None,causal_tower_height=4,gamma=1/5):
    super(SyncNet,self).__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tower_filters = tower_filters if tower_filters else [56,96,96,90,96,108,96]
    self.num_towers = num_towers
    self.gamma = gamma
    self.samples_per_window = 0
    self.shapes = []
    setattr(self, 'causal_ref', nn.ModuleList([CausalTower(tower_filters=self.tower_filters[0:causal_tower_height]) for tower in range(num_towers)]))
    setattr(self, 'causal_inp', nn.ModuleList([CausalTower(tower_filters=self.tower_filters[0:causal_tower_height]) for tower in range(num_towers)]))
    self.tower_height = causal_tower_height
    self.windows = None
    num_anticausal = num_towers
    anti_causal_filters=[[(12,8),(8,8)],[(8,6),(6,6)],[(6,4),(4,4)],[(4,2),(2,2)],[(2,1),(1,1)]]#[[(16,12),(12,8)],[(8,6),(6,4)],[(4,2),(2,1)]]
    for anti_causal in range(len(self.tower_filters)-self.tower_height):
      filter=int((2/3)*(self.tower_filters[anti_causal+self.tower_height]))

      num_anticausal -= 3
      #print(num_anticausal)
      setattr(self,
              'anticausal_{}_ref'.format(anti_causal+1),
              nn.ModuleList([nn.Sequential(*[nn.Conv1d(in_channels=anti_causal_filters[anti_causal][0][0],
                                                       out_channels=anti_causal_filters[anti_causal][0][1],
                                                       kernel_size=self.tower_filters[anti_causal+self.tower_height],dilation=5 if anti_causal in [0,1,2] else 4),\
                                             nn.ReLU(),\
                                             nn.Conv1d(in_channels=anti_causal_filters[anti_causal][1][0],
                                                       out_channels=anti_causal_filters[anti_causal][1][1],
                                                       kernel_size=filter,bias=False,dilation=5 if anti_causal in [0,1,2] else 4),\
                                             nn.BatchNorm1d(num_features=anti_causal_filters[anti_causal][1][1]),\
                                             nn.ReLU()]) for i in range(num_anticausal)]))
      setattr(self,
              'anticausal_{}_inp'.format(anti_causal+1),
              nn.ModuleList([nn.Sequential(*[nn.Conv1d(in_channels=anti_causal_filters[anti_causal][0][0],
                                                       out_channels=anti_causal_filters[anti_causal][0][1],
                                                       kernel_size=self.tower_filters[anti_causal+self.tower_height],dilation=5 if anti_causal in [0,1,2] else 4),\
                                             nn.ReLU(),\
                                             nn.Conv1d(in_channels=anti_causal_filters[anti_causal][1][0],
                                                       out_channels=anti_causal_filters[anti_causal][1][1],
                                                       kernel_size=filter,bias=False,dilation=5 if anti_causal in [0,1,2] else 4),\
                                             nn.BatchNorm1d(num_features=anti_causal_filters[anti_causal][1][1]),\
                                             nn.ReLU()]) for i in range(num_anticausal)]))  

  def _get_anti_causal(self,y_dash,shapes,id=0):
    tower_id=1; win=shapes[0]; result_ref=torch.Tensor().to(self.device);result_inp=torch.Tensor().to(self.device); shape=[]#; ppp=1
    while True:
        if tower_id+3==len(shapes):
          third = F.pad(input=y_dash[:,:,win+shapes[tower_id]+shapes[tower_id+1]:win+shapes[tower_id]+shapes[tower_id+1]+shapes[tower_id+2]],
                      pad=(0, abs(shapes[tower_id+2]-shapes[tower_id+1])), mode='constant', value=0)
        else:
          third = y_dash[:,:,win+shapes[tower_id]+shapes[tower_id+1]:win+shapes[tower_id]+shapes[tower_id+1]+shapes[tower_id+2]]
        segment = y_dash[:,:,win:win+shapes[tower_id]] + y_dash[:,:,win+shapes[tower_id]:win+shapes[tower_id]+shapes[tower_id+1]] + third
        layer_ref = getattr(self,'anticausal_{}_ref'.format(str(id+1)))[tower_id-1]
        layer_inp = getattr(self,'anticausal_{}_inp'.format(str(id+1)))[tower_id-1];
        slice_at = int(segment.shape[1]/2); 
        output_ref = layer_ref(segment[:,0:slice_at,:]);output_inp=layer_inp(segment[:,slice_at:,:])
        shape.append(output_ref.shape[2])
        result_ref = torch.cat((result_ref,output_ref),dim=2)
        result_inp = torch.cat((result_inp,output_inp),dim=2)
        win += shapes[tower_id];
        tower_id += 1
        if tower_id+2==len(shapes):
          break
    return torch.cat((result_ref,result_inp),dim=1),shape
    
  def _update_gamma(self):
    pass

  def _find_windows(self,signal):
    self._update_gamma()
    self.samples_per_window = np.floor(signal.shape[2]/((1 - self.gamma)*(self.num_towers)))

  def forward(self,signal):
    reference = torch.unsqueeze(signal[:,0,:],dim=1)
    input = torch.unsqueeze(signal[:,1,:],dim=1)
    self.shapes=[]
    y_ref,y_inp = torch.Tensor().to(self.device),torch.Tensor().to(self.device)
    self._find_windows(signal=signal)
    w0 = 0; delta = np.ceil((self.samples_per_window)*(self.gamma)); w1 = delta
    for towerid in range(self.num_towers):
      w0 = int(w1-delta)
      w1 = int(w0+self.samples_per_window-1)
      if towerid==0:
        tower_output_ref,tower_output_inp={},{}
        for l in range(self.tower_height + 1):
          tower_output_ref['layer'+str(l)]=0
          tower_output_inp['layer'+str(l)]=0
      getattr(self,'causal_ref')[towerid].signal = reference
      getattr(self,'causal_inp')[towerid].signal = input
      tower_output_ref = getattr(self,'causal_ref')[towerid](window=(w0,w1),prev_tower_outputs=tower_output_ref)
      tower_output_inp = getattr(self,'causal_inp')[towerid](window=(w0,w1),prev_tower_outputs=tower_output_inp); 
      self.shapes.append(tower_output_inp['layer'+str(self.tower_height)].shape[2])
      y_ref = torch.cat((y_ref,tower_output_ref['layer'+str(self.tower_height)]),dim=2)
      y_inp = torch.cat((y_inp,tower_output_inp['layer'+str(self.tower_height)]),dim=2); 
    result = torch.cat((y_ref,y_inp),dim=1); 
    for anti_causal in range(len(self.tower_filters)-self.tower_height):
      result,shapes = self._get_anti_causal(y_dash=result,
                                            shapes=self.shapes if anti_causal==0 else shapes,
                                            id=anti_causal)
    return result


class hybrid_loss(nn.Module):

  def __init__(self,l1=0.8,l2=1.2,l3=1):
    super(hybrid_loss,self).__init__()
    self.mse = nn.MSELoss(reduction='none')
    self.kl = nn.KLDivLoss(reduction='none',log_target=False)
    self.hp={'corr':l1,'pool1':l2,'pool2':l3}
    self.pool=nn.MaxPool1d(kernel_size=163,stride=163,padding=0,return_indices=False)

  def forward(self,predicted_corr,target_corr,target_pool,weights_corr,pool_weights):
    pool_weights = supp.normalized(pool_weights)+0.05
    corr_loss = self.hp['corr']*weights_corr*torch.sqrt(self.mse(torch.log(predicted_corr+0.001),torch.log(target_corr+0.001))+0.0001)
    predicted_corr = self.pool(predicted_corr)
    pooled_loss1 = self.hp['pool1']*pool_weights*self.mse(predicted_corr,target_pool)
    pooled_loss2 = self.hp['pool2']*pool_weights*self.kl(F.log_softmax(predicted_corr,dim=1),F.softmax(target_pool,dim=1))
    return corr_loss.mean()+pooled_loss1.mean()+pooled_loss2.mean()