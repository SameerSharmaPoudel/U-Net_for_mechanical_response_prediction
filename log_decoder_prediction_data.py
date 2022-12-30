import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd
from load_config_file import load_config
import sys
import os
sys.path.append('models/')
from models import model_list, model_name

config = load_config('config1.yaml')

def log_dec_pred_image_data(epoch,path,image,decoded,target,train=True):
    """
    stores input microstructure images (original_binary_images), predicted images from decoder (decoded_images), ground truth images (targets) in hdf5 format for both training and validation phases 
    """ 
    filename = os.path.join(path, 'decoder_prediction.h5')
    F = h5py.File(filename, 'a')
    decoded = decoded.detach()
    
    if train==True:
            
        group1 = F.require_group('dec_pred_training_image_data')
        
        subgroup1 = group1.require_group('Epoch_{}'.format(epoch))
        dset_11 = subgroup1.require_dataset('original_binary_images', data= image.cpu(), shape=image.cpu().shape, dtype=np.float64, compression='gzip')
        dset_12 = subgroup1.require_dataset('decoded_images', data= decoded.cpu(), shape=decoded.cpu().shape, dtype=np.float64, compression='gzip')
        dset_13 = subgroup1.require_dataset('targets', data= target.cpu(), shape=target.cpu().shape, dtype=np.float64, compression='gzip')
    
    if train==False:
           
        group4 = F.require_group('dec_pred_validation_image_data')
		
        subgroup4 = group4.require_group('Epoch_{}'.format(epoch))		
        dset_41 = subgroup4.require_dataset('original_binary_images', data= image.cpu(), shape=image.cpu().shape, dtype=np.float64, compression='gzip')
        dset_42 = subgroup4.require_dataset('decoded_images', data= decoded.cpu(), shape=decoded.cpu().shape, dtype=np.float64, compression='gzip')
        dset_43 = subgroup4.require_dataset('targets', data= target.cpu(), shape=target.cpu().shape, dtype=np.float64, compression='gzip')		    
    
    F.close()
    print('Decoder Prediction Image Data Stored !!')
	
	
def log_dec_pred_loss_and_Rsquare_score(training_mse_loss, val_mse_loss, training_r_square_xx, validation_r_square_xx, training_r_square_xy, validation_r_square_xy, path):
    """
    stores training and validation losses, and r-squared score training and validation accuracies as lists for the prediction from decoder in hdf5 format 
    """
    filename = os.path.join(path, 'decoder_prediction.h5')
    F = h5py.File(filename, 'a')
  
    group2 = F.require_group('dec_pred_training_loss')    
    dset_21 = group2.require_dataset('training_loss', data= training_mse_loss, shape=np.shape(training_mse_loss), dtype=np.float64, compression='gzip')

    group3 = F.require_group('dec_pred_training_r_square_score')    
    dset_31 = group3.require_dataset('training_r_square_score_xx', data= training_r_square_xx, shape=np.shape(training_r_square_xx), dtype=np.float64, compression='gzip')
    dset_32 = group3.require_dataset('training_r_square_score_xy', data= training_r_square_xy, shape=np.shape(training_r_square_xy), dtype=np.float64, compression='gzip')
									
    group5 = F.require_group('dec_pred_validation_loss')    
    dset_51 = group5.require_dataset('validation_loss', data= val_mse_loss, shape=np.shape(val_mse_loss), dtype=np.float64, compression='gzip')
    	
    group6 = F.require_group('dec_pred_validation_r_square_score')    
    dset_61 = group6.require_dataset('validation_r_square_score_xx', data= validation_r_square_xx, shape=np.shape(validation_r_square_xx), dtype=np.float64, compression='gzip') 
    dset_62 = group6.require_dataset('validation_r_square_score_xy', data= validation_r_square_xy, shape=np.shape(validation_r_square_xy), dtype=np.float64, compression='gzip')
    
    F.close()	
    print('Decoder Prediction Loss and R_square Score Stored!!!')