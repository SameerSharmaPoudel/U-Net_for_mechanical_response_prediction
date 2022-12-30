import h5py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import mean
import torch

def compute_hom_flux(bin_img, dec_pred_xx, dec_target_xx, dec_pred_xy, dec_target_xy):
    
    """
    Compute the homogenized heat flux responses.
    
    INPUT(s):
        bin_img          : input microstructure binary image(torch.tensor), dimension: batch_size * height * width
        dec_pred_xx      : predicted heat flux component in x direction(torch.tensor), dimension: batch_size * height * width
        dec_pred_xy     : predicted heat flux component in y direction(torch.tensor), dimension: batch_size * height * width
        dec_target_xx   : ground truth heat flux component in x direction(torch.tensor), dimension: batch_size * height * width
        dec_target_xy   : ground truth heat flux component in y direction(torch.tensor), dimension: batch_size * height * width

    OUTPUT(s): 
        rae_total_batch      : relative absolute error of the overall homogenized prediction 
                                for all samples in a batch (stored in a list)
        rae_incl_batch      : relative absolute error of the homogenized prediction from inclusions
                                for all samples in a batch (stored in a list)
        rae_matr_batch     : relative absolute error of the homogenized prediction from matrix
                                for all samples in a batch (stored in a list)

"""
    
    bin_img = bin_img.cpu().detach().numpy()
    dec_pred_xx = dec_pred_xx.cpu().detach().numpy()
    dec_target_xx = dec_target_xx.cpu().detach().numpy()
    dec_pred_xy = dec_pred_xy.cpu().detach().numpy()
    dec_target_xy = dec_target_xy.cpu().detach().numpy()
    
    rae_total_batch = []
    rae_incl_batch = []
    rae_matr_batch = []

    for b in range(len(bin_img)):
                                
        dec_pred_xx_mean = np.mean(dec_pred_xx[b,:,:])
        dec_pred_xy_mean = np.mean(dec_pred_xy[b,:,:])

        dec_target_xx_mean = np.mean(dec_target_xx[b,:,:])
        dec_target_xy_mean = np.mean(dec_target_xy[b,:,:])
        
        chi_1 = bin_img[b,:,:]
        chi_0 =  1 - bin_img[b,:,:]
             
        dec_pred_xx_incl_mean = np.mean(dec_pred_xx[b,:,:] * chi_1) / np.mean(chi_1)
        dec_pred_xx_matr_mean = np.mean(dec_pred_xx[b,:,:] * chi_0) / np.mean(chi_0)
                
        dec_target_xx_incl_mean = np.mean(dec_target_xx[b,:,:] * chi_1) / np.mean(chi_1) 
        dec_target_xx_matr_mean = np.mean(dec_target_xx[b,:,:] * chi_0) / np.mean(chi_0)
              
        dec_pred_xy_incl_mean = np.mean(dec_pred_xy[b,:,:] * chi_1) / np.mean(chi_1)
        dec_pred_xy_matr_mean = np.mean(dec_pred_xy[b,:,:] * chi_0) / np.mean(chi_0)
        
        dec_target_xy_incl_mean =  np.mean(dec_target_xy[b,:,:] * chi_1) / np.mean(chi_1)
        dec_target_xy_matr_mean =  np.mean(dec_target_xy[b,:,:] * chi_0) / np.mean(chi_0)

        q_error = np.array([dec_target_xx_mean - dec_pred_xx_mean, dec_target_xy_mean - dec_pred_xy_mean])
        q_error_norm = np.linalg.norm(q_error, ord=1)
        q_avg = np.array([dec_target_xx_mean, dec_target_xy_mean])
        q_avg_norm = np.linalg.norm(q_avg, ord=1)
        rae_total = q_error_norm/q_avg_norm
        
        q_error_incl = np.array([dec_target_xx_incl_mean - dec_pred_xx_incl_mean, dec_target_xy_incl_mean - dec_pred_xy_incl_mean])
        q_error_incl_norm = np.linalg.norm(q_error_incl, ord=1)
        q_avg_incl = np.array([dec_target_xx_incl_mean, dec_target_xy_incl_mean])
        q_avg_incl_norm = np.linalg.norm(q_avg_incl, ord=1)
        rae_incl = q_error_incl_norm/q_avg_incl_norm
        
        q_error_matr = np.array([dec_target_xx_matr_mean - dec_pred_xx_matr_mean, dec_target_xy_matr_mean - dec_pred_xy_matr_mean])
        q_error_matr_norm = np.linalg.norm(q_error_matr, ord=1)
        q_avg_matr = np.array([dec_target_xx_matr_mean, dec_target_xy_matr_mean])
        q_avg_matr_norm = np.linalg.norm(q_avg_matr, ord=1)
        rae_matr = q_error_matr_norm/q_avg_matr_norm
               
        rae_total_batch.append(rae_total)
        rae_incl_batch.append(rae_incl)
        rae_matr_batch.append(rae_matr)
         
    return rae_total_batch, rae_incl_batch, rae_matr_batch





