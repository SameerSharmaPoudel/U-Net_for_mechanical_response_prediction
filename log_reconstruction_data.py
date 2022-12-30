import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd
from load_config_file import load_config
import sys
import os
sys.path.append('models/')
from models import model_list, model_name

config = load_config('config1.yaml')

def log_recon_image_data(epoch,path,image,encoded_image,reconstructed_image,train=True):
    """
    stores input microstructure images (original_binary_images), reconstructed images, compressed images in latent space (encoded_images), 
    and covariance matrix, pearson correlation coefficient and svd of the encoded images in hdf5 format for both training and validation phases 
    """    
    filename = os.path.join(path, 'reconstruction.h5')
    F = h5py.File(filename, 'a')
    
    encoded_image = encoded_image.cpu().detach().numpy() 
    encoded_image = encoded_image[:,:,0,0]
    #print(np.shape(encoded_image))
    reconstructed_image = reconstructed_image.detach()
    covariance_matrix = np.cov(encoded_image,rowvar=False)
    p_corr_coeff_matrix = np.corrcoef(covariance_matrix,rowvar=False)
    V, S, W_T = randomized_svd(encoded_image, n_components=128) #the min of encoded_image dimension is less than n_components
    
    if train==True:
            
        group1 = F.require_group('recon_training_image_data')
        
        subgroup1 = group1.require_group('Epoch_{}'.format(epoch))
        dset_11 = subgroup1.require_dataset('original_binary_images', data= image.cpu(), shape=image.cpu().shape, dtype=np.float64, compression='gzip')
        dset_12 = subgroup1.require_dataset('reconstructed_images', data= reconstructed_image.cpu(), shape=reconstructed_image.cpu().shape, dtype=np.float64, compression='gzip')
        dset_13 = subgroup1.require_dataset('encoded_image', data= encoded_image, shape=encoded_image.shape, dtype=np.float64, compression='gzip')
        dset_14 = subgroup1.require_dataset('covariance_matrix', data= covariance_matrix, shape=covariance_matrix.shape, dtype=np.float64, compression='gzip')
        dset_15 = subgroup1.require_dataset('p_corr_coeff_matrix', data= p_corr_coeff_matrix, shape=p_corr_coeff_matrix.shape, dtype=np.float64, compression='gzip')
        dset_16 = subgroup1.require_dataset('svd', data= S, shape=np.shape(S), dtype=np.float64, compression='gzip') 
   
    if train==False:
        
    
        group4 = F.require_group('recon_validation_image_data')
		
        subgroup4 = group4.require_group('Epoch_{}'.format(epoch))		
        dset_41 = subgroup4.require_dataset('original_binary_images', data= image.cpu(), shape=image.cpu().shape, dtype=np.float64, compression='gzip')
        dset_42 = subgroup4.require_dataset('reconstructed_images', data= reconstructed_image.cpu(), shape=reconstructed_image.cpu().shape, dtype=np.float64, compression='gzip')    
        dset_43 = subgroup4.require_dataset('encoded_image', data= encoded_image, shape=encoded_image.shape, dtype=np.float64, compression='gzip')
        dset_44 = subgroup4.require_dataset('covariance_matrix', data= covariance_matrix, shape=covariance_matrix.shape, dtype=np.float64, compression='gzip')
        dset_45 = subgroup4.require_dataset('p_corr_coeff_matrix', data= p_corr_coeff_matrix, shape=p_corr_coeff_matrix.shape, dtype=np.float64, compression='gzip')
        dset_46 = subgroup4.require_dataset('svd', data= S, shape=np.shape(S), dtype=np.float64, compression='gzip')
		   
    F.close()
    print('Reconstruction Image Data Stored !!')
	
	
def log_recon_loss_and_Rsquare_score(training_mse_loss, val_mse_loss, training_r_square, validation_r_square, path):
    """
    stores reconstruction training and validation losses, and r-squared score training and validation accuracies as lists in hdf5 format 
    """
    filename = os.path.join(path, 'reconstruction.h5')
    F = h5py.File(filename, 'a')
    	
    group2 = F.require_group('recon_training_loss')    
    dset_21 = group2.require_dataset('training_loss', data= training_mse_loss, shape=np.shape(training_mse_loss), dtype=np.float64, compression='gzip')
    	
    group3 = F.require_group('recon_training_r_square_score')    
    dset_31 = group3.require_dataset('training_r_square_score', data= training_r_square, shape=np.shape(training_r_square), dtype=np.float64, compression='gzip')
    									
    group5 = F.require_group('recon_validation_loss')    
    dset_51 = group5.require_dataset('validation_loss', data= val_mse_loss, shape=np.shape(val_mse_loss), dtype=np.float64, compression='gzip')
    	
    group6 = F.require_group('recon_validation_r_square_score')    
    dset_61 = group6.require_dataset('validation_r_square_score', data= validation_r_square, shape=np.shape(validation_r_square), dtype=np.float64, compression='gzip') 
    
    F.close()	
    print('Reconstruction Loss and R_square Score Stored!!!')