import sys
sys.path.append('models/')
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from load_config_file import load_config
from model_utils import feed_model_to_device
from models_test import model_list, model_name
from eval_metrics import compute_error_metrics, compute_r_square
import dataloader_xy
import time
from get_hom_flux import compute_hom_flux
import os
import csv
import h5py
import numpy as np
###############################################################################
#                         Test on Unseen data 
"""
    This script evaluates the best models on unseen test data based on:
        (a) mse
        (b) mae
        (c) prediction time
        (d) r-squared score accuracy
        
        and also stores cumulative relative absolute error of overall homogenized
        prediction (including both inclusions and matrix), homogenized prediction 
        from inclusions only and homogenized prediction from matrix only,
        from the best models, for all samples in the test dataset
        
"""
###############################################################################

config = load_config('config1.yaml')

class my_dataset(): 
    
    def __init__(self, config):

        self.config = config
        self.F = h5py.File(self.config['Dataset']['test_data_file_path'],'r')
                   
    def __getitem__(self, index):
               
        image_path = self.config['Dataset']['image_path']
        image = self.F[image_path.format(index+1)][:]
        image = np.expand_dims(image,axis=(0,1))
        
        target_path_xx = self.config['Dataset']['target_path_xx']
        target_xx = self.F[target_path_xx.format(index+1)][:]
        target_xx = np.expand_dims(target_xx,axis=(0,1))
        
        target_path_xy = self.config['Dataset']['target_path_xy']
        target_xy = self.F[target_path_xy.format(index+1)][:]
        target_xy = np.expand_dims(target_xy,axis=(0,1)) 
        
        return (image,target_xx,target_xy)
    
    def __len__(self):

        return len(self.F['image_data'])
            
dataset = my_dataset(config)

path_best_models  = [path_all_256, path_all_512, path_all_1024, path_all_2048, path_all_16x16, path_all_32x32]

parent_dir = 'new_data/results_test'
for model_idx in range(len(model_list)):

    model = model_list[model_idx]
    mn = model_name[model_idx]
    device = torch.device('cuda:0') 
    
    model_path = path_best_models[model_idx]
        
    for path_type in range(len(path_all_256)):
        if path_type==0:
            name ='recon'
        elif path_type==1:
            name='decoder'
        elif path_type==2:
            name='whole_model'
            
        directory = mn + '_' + name
        save_path = os.path.join(parent_dir, directory)
        try:
            os.makedirs(save_path, exist_ok = True)
            print("Directory '%s' created successfully" %directory)
        except OSError as error:
            print("Directory '%s' can not be created")
        
        PATH = model_path[path_type]
        
        loaded_checkpoint = torch.load(PATH)
        epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_state'])
        model = model.to(device)
        
        optimizer = Adam(model.parameters(), config['Hyperparameters']['lr'])
        optimizer.load_state_dict(loaded_checkpoint['optim_state'])
                
        with torch.no_grad():
            
            model.eval()
            
            running_r_square = 0.0
            running_mae = 0.0
            running_mse = 0.0
            
            cum_rae = []
            cum_rae_incl = []
            cum_rae_matr = []
        
            for i in range(len(dataset)):
                                 
                image, target_xx, target_xy = dataset[i]
                image, target_xx, target_xy = torch.from_numpy(image), torch.from_numpy(target_xx), torch.from_numpy(target_xy)
                image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
                            
                start_time = time.time()
                predicted = model(image)
                end_time = time.time()
                total_time = end_time - start_time
                
                predicted_xx, predicted_xy = predicted[:,0,:,:],  predicted[:,1,:,:]
                
                rae_total, rae_incl, rae_matr = compute_hom_flux(image, predicted_xx, target_xx, predicted_xy, target_xy)
                cum_rae.extend(rae_total)
                cum_rae_incl.extend(rae_incl)
                cum_rae_matr.extend(rae_matr)
                
                if path_type==0:
                    r_square =  compute_r_square(image, predicted_xx)   
                    mae, mse =  compute_error_metrics(image, predicted_xx)                    
                else:
                    r_square_xx =  compute_r_square(target_xx, predicted_xx)  
                    r_square_xy =  compute_r_square(target_xy, predicted_xy)                    
                    mae_xx, mse_xx =  compute_error_metrics(target_xx, predicted_xx)  
                    mae_xy, mse_xy =  compute_error_metrics(target_xy, predicted_xy)
                    r_square = 0.5*(r_square_xx + r_square_xy)
                    mae = 0.5*(mae_xx + mae_xy)
                    mse = 0.5*(mse_xx + mse_xy)
                                    
                running_r_square += r_square
                running_mse += mse
                running_mae += mae
                  
            running_r_square = running_r_square/len(dataset)
            running_mse = running_mse/len(dataset)
            running_mae = running_mae/len(dataset)
                    
            print('r_square:', r_square)
            print('mse:', mse)
            print('mae:', mae)
            print('time  for prediction:', total_time)

            header = ['R_Square', 'MSE', 'MAE', 'Prediction Time']
            data = [[r_square, mse, mae, total_time]]
            filename = os.path.join(save_path, 'metrics.csv')
            with open(filename, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
            print('Metrics Stored!')
                
            filename = os.path.join(save_path, 'cum_rae.h5')
            F = h5py.File(filename, 'a')            
            group1 = F.require_group('total_rae')    
            dset_11 = group1.require_dataset('total_rae', data= cum_rae, shape=np.shape(cum_rae), dtype=np.float64, compression='gzip')
            group2 = F.require_group('rae_incl')    
            dset_21 = group2.require_dataset('rae_incl', data= cum_rae_incl, shape=np.shape(cum_rae_incl), dtype=np.float64, compression='gzip')
            group3 = F.require_group('rae_matr')    
            dset_31 = group3.require_dataset('rae_matr', data= cum_rae_matr, shape=np.shape(cum_rae_matr), dtype=np.float64, compression='gzip')
            F.close()	
            print('Cumulative RAE stored!')
            
            cum_rae.clear()
            cum_rae_incl.clear()
            cum_rae_matr.clear()

print('ALL NECESSARY DATA STORED!!!')
