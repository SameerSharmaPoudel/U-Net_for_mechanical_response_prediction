import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from load_config_file import load_config
import random

config = load_config('config1.yaml') 

class my_dataset(): 
    
    def __init__(self, config):

        self.config = config
        self.F = h5py.File(self.config['Dataset']['data_file_path'],'r')
                   
    def __getitem__(self, index):
               
        image_path = self.config['Dataset']['image_path']
        image = self.F[image_path.format(index+1)][:]
        image = np.expand_dims(image,axis=0)
        
        target_path_xx = self.config['Dataset']['target_path_xx']
        target_xx = self.F[target_path_xx.format(index+1)][:]
        target_xx = np.expand_dims(target_xx,axis=0)
        
        target_path_xy = self.config['Dataset']['target_path_xy']
        target_xy = self.F[target_path_xy.format(index+1)][:]
        target_xy = np.expand_dims(target_xy,axis=0)
        
        return (image,target_xx,target_xy)
    
    def __len__(self):

        return len(self.F['image_data'])
    
        
dataset = my_dataset(config)

test_amount  = int(len(dataset) * config['Dataset']['test_percentage'])
val_amount = int(len(dataset) * config['Dataset']['val_percentage'])                  
train_amount = int(len(dataset) - (test_amount + val_amount))

train_set, val_set, test_set = random_split(dataset, [train_amount, val_amount,	test_amount])	

train_dataloader = DataLoader(train_set, config['Hyperparameters']['train_batch_size'], shuffle=True, pin_memory=False, drop_last=  True)
val_dataloader = DataLoader(val_set, config['Hyperparameters']['val_batch_size'], pin_memory=False, drop_last=  True)
test_dataloader = DataLoader(test_set, config['Hyperparameters']['test_batch_size'], shuffle= False, pin_memory=False, drop_last=  True)


    