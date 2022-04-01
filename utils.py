import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
import os
import glob

class synthetic_data:
  '''
  Generate synthetic audio files and save into given dir.
  '''
  
  def __init__(self,num_signals=200,noise=True,snr=None,fundamental_period=1):

    self.num_signals = num_signals
    self.noise = noise
    self.snr = snr if snr else [-10,20]
    self.max_tau = fundamental_period
    self.delay_choices = np.arange(0.09, self.max_tau, self.max_tau/500)
  
  def write_files(self,
                  reference_signal,
                  dir_to_save):    
    ref_signal,sr = torchaudio.load(reference_signal)
    annotations = pd.DataFrame(np.zeros((self.num_signals,3)),columns=['FileNames', 'Delays', 'SNR(dB)'])
    if not os.path.exists(dir_to_save):
      os.mkdir(dir_to_save)
    print('Writing synthetic audio files to the {} directory'.format(dir_to_save.split('/')[-1]))
    for sig_id in range(self.num_signals):
      synth_signal, delay, snr = self.generate_audio(ref_signal=ref_signal,sr=sr)
      signal_path = dir_to_save+'/sig'+str(sig_id+1)+'.wav'
      torchaudio.save(signal_path, synth_signal, sr)
      annotations.iloc[sig_id,:] = [signal_path,delay,snr]
    with pd.ExcelWriter(dir_to_save + '/annotations.xlsx') as writer:
      annotations.to_excel(writer)
    print('Writing Finished')
    return annotations

  def generate_audio(self,ref_signal,sr=16000):
    synth_signal, delay = self._shift_signal(signal=ref_signal,sr=sr)
    if self.noise:
      synth_signal, snr = self._inject_noise(signal=synth_signal)
    else:
      snr=None
    return synth_signal,delay,snr
        
  def _inject_noise(self,signal):
    P_signal=torch.mean(signal**2)
    snr=np.random.uniform(self.snr[0],self.snr[1])
    P_noise = P_signal/(10**(snr/10))
    noise=torch.Tensor(np.random.normal(loc=0,scale=np.sqrt(P_noise),size=signal.shape))
    P_noise=torch.mean(noise**2)
    snr=10*np.log10(P_signal/P_noise)
    synth=(signal+noise)
    return synth/synth.max(),snr.item()
  
  def _shift_signal(self,signal,sr):
    shift=np.random.choice(self.delay_choices)*sr
    signal=torch.roll(signal,int(shift))
    signal[0,0:int(shift)]=0
    return signal, shift/sr


class prepare_data(Dataset):
  '''
  Custom PyTorch Dataset, reading original/synthetic/both types of audio files.
  '''
   
  def __init__(self,
               m=None,
               csv='/data/M_Tic/Data-1.xlsx',
               transform=None,
               include='both',
               overwrite=True,
               synthetic_csv_file='/data/M_Tic/Synthetic/annotations.xlsx',
               num_signals=400,
               noise=True,
               snr=None,
               train=0.85): 
    cwd = os.getcwd()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.train = True
    self.transform = transform
    orig_delays = pd.read_excel(os.path.join(cwd,csv),header=None,names=['FileNames','Delays'],skiprows=1,usecols=(4,5))
    orig_delays.drop_duplicates(subset='FileNames',inplace=True,ignore_index=True)
    path = os.path.join(cwd,'/data/M_Tic')
    lst = glob.glob(path+'/audio/*.wav')
    annotations = pd.DataFrame(np.zeros((len(lst),3)),columns=['FileNames','Delays','SNR(dB)'])
    self.reference,_ = torchaudio.load(os.path.join(cwd,'/data/M_Tic/reference.wav')) 
    self.reference = torch.squeeze(self.reference,dim=0)
    if self.transform:
      self.reference=self.transform(self.reference)
    for id,audio in enumerate(lst):
      index = orig_delays['FileNames']==lst[id-1].split('/')[-1]
      delay = orig_delays['Delays'][index].item()
      annotations.iloc[id,:] = [audio,delay,0]
    if include in ['both','synthetic']:
      self.noise = noise
      self.snr = snr if snr else [-20,60]
      if overwrite:
        synth = synthetic_data(num_signals=num_signals,noise=self.noise,snr=self.snr)
        annotations_synth = synth.write_files(reference_signal=os.path.join(cwd,'/data/M_Tic/reference.wav'),
                                              dir_to_save=os.path.join(cwd,'/data/M_Tic/Synthetic'))
      else:
        annotations_synth = pd.read_excel(os.path.join(cwd,synthetic_csv_file),header=None,skiprows=1,usecols=(1,2,3),names=['FileNames','Delays','SNR(dB)'])
    if include == 'both':
      annotations = pd.concat([annotations, annotations_synth])
    elif include == 'synthetic':
      annotations = annotations_synth.copy()
    annotations = annotations.sort_values(by='Delays',ignore_index=True)
    sig = torch.stack((self.reference,self.reference),dim=0)
    with torch.no_grad():
      self.length_pred = m(torch.unsqueeze(sig,dim=0).to(self.device)).shape[2]
    annotations = annotations.sample(frac=1,random_state=11).reset_index(drop=True)
    self.annotations_train = annotations.iloc[0:int(train*(num_signals+len(lst))),:]
    self.annotations_test = annotations.iloc[int(train*(num_signals+len(lst))):,:]
  
  def _generate_target(self,tau,period=16000,std=1700):
    target = torch.Tensor(); mean=tau
    num_gauss = np.ceil(self.length_pred/period)
    for gauss_id in range(int(num_gauss)):
      x=torch.linspace(0,self.length_pred,self.length_pred)
      y=norm.pdf(x,mean,std)
      mean += period
      if gauss_id==0:
        target=y.copy()
      else:
        target+=y
    return torch.Tensor(target)

  def _get_pooled_targets(self,delay_in_samples):
    length = int(self.length_pred/163)
    arr = torch.full((length,),0.001)
    arr[int(delay_in_samples/163)] = 1
    return arr

  def _get_weights_for_correlation(self,target):
    ones_idx = np.where(target==target.max())[0]
    id=0; div=10;length=len(target)
    while True:
      res=length/div
      if np.floor(res)==0:
        break
      id+=1
      div*=10
    m = 1 - (len(ones_idx)/div)
    k = m + (length/div); 
    weights = torch.full((target.shape[0],),m)
    ones_idx = np.where(target==target.max())[0]
    weights = weights.index_fill(0,torch.Tensor(ones_idx).long(),value=1.2*k)
    return weights

  def _get_weights_for_pool(self,delay):
    x = torch.linspace(0,98,98)
    array = torch.Tensor(norm.pdf(x,int(delay/163),5))
    return array

  def __len__(self):
    if self.train:
      return len(self.annotations_train)
    else:
      return len(self.annotations_test)

  def __getitem__(self,index):
    if self.train:
      filename,delay,snr = self.annotations_train.iloc[index,:]
    else:
      filename,delay,snr = self.annotations_test.iloc[index,:]
    signal, sr = torchaudio.load(filename)
    signal = torch.squeeze(signal,dim=0)
    if self.transform:
      signal=self.transform(signal.to(self.device))
    if signal.shape[0]<100000:
      signal=torch.clone(self.reference)
    if self.reference.shape[0]>signal.shape[0]:
      signal=F.pad(input=signal,pad=(0,self.reference.shape[0]-signal.shape[0]), mode='constant', value=0)
      result=torch.stack((self.reference,signal),dim=0)
    else:
      result=torch.stack((self.reference,signal[0:self.reference.shape[0]]),dim=0)
    true=self._generate_target(tau=int(delay*sr),period=sr,std=1350)
    return {'input_sequence':result,
            'target_sequence_corr':true,
            'target_sequence_pooled':self._get_pooled_targets(delay_in_samples=int(delay*sr)),
            'weights_corr':self._get_weights_for_correlation(true),
            'weights_pool':self._get_weights_for_pool(int(delay*sr))}

class supplementary:
    '''
    Some helping functions
    '''
    
    def __init__(self):
        pass
    
    def count_parameters(self,model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def normalized(self,tensor):
      batches,samples=tensor.shape
      max = torch.unsqueeze(torch.max(tensor,dim=1)[0],dim=1) + 0.00001
      divided = torch.divide(tensor,max)
      return divided
  
    def correlation(self,signal):
      in1 = self.normalized(signal[:,0,:])
      in2 = self.normalized(signal[:,1,:])
      n = in1.shape[1]+in2.shape[1]-1
      fft_1 = torch.fft.rfft(in1, n=n)
      fft_2 = torch.fft.rfft(-in2, n=n)
      fft_multiplied = fft_1*fft_2
      prelim_correlation = abs(torch.fft.irfft(fft_multiplied, n=n))
      return self.normalized(prelim_correlation[:, 0:int((n+1)/2)])