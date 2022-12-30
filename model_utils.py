import torch
from torch.nn import DataParallel
from load_config_file import load_config
import numpy as np

config = load_config('config1.yaml')

def feed_model_to_device(model, config):
    
    """
    selects the available gpu and
    returns the model after loading it in the selected gpu and the selected gpu
    """
    
    if torch.cuda.is_available():
        
        no_GPUs = torch.cuda.device_count()
        print(f'Number of GPUs available:{no_GPUs}')
                
        if config['Device']['all_GPUs'] == 'Yes': 
    
            print('Running on', no_GPUs, 'GPUs!')
            
            device = torch.device('cuda')
            model = DataParallel(model)
            model.to(device)
            
        else:                 
            GPU_list = [i for i in range(no_GPUs)]
            GPU_idx = config['Device']['select_GPU']
            
            if GPU_idx in GPU_list:
                print('Running on GPU:',GPU_idx)                
                device = torch.device('cuda:{}'.format(GPU_idx))
                model.to(device)
                     
    else:
        device = torch.device('cpu')
        print('Running on the CPU')
        model.to(device)    
        
    return model, device

def roll_data(images, targets_xx, targets_xy, part=0.50, shuffle=False):
    """
    ## Note that this function is executed on the CPU with numpy.
    It performs data augmentation using translation technique (roll).
    
    INPUT(s):
    -----------
    images:        input microstructure binary images (torch.tensor), dimension: batch_size * no_of_channels * height * width
    targets_xx:    ground truth heat flux component in x direction(torch.tensor), dimension: batch_size * no_of_channels * height * width
    targets_xy:    ground truth heat flux component in y direction(torch.tensor), dimension: batch_size * no_of_channels * height * width     
    part:         float, default 0.5
                  what proportion of the randomly selected images should be rolled
    shuffle:     bool, default False
                if the data should be shuffled during augmentation
    
    OUTPUT(s):
    --------
    images:        numpy array of images with randomly selected <part> randomly rolled
    targets_xx:   numpy array of targets_xx with randomly selected <part> randomly rolled
    targets_xy:   numpy array of targets_xy with randomly selected <part> randomly rolled
    
    """
        
    n_batch = images[0].shape[0]
    n_data_aug = int(n_batch*part)
    n_roll = n_data_aug
    img_dim = images.shape[2:]
    max_trans = min(img_dim)
    indices = np.random.permutation(n_batch)
    trans = np.random.randint(0, max_trans, size=(n_data_aug, len(img_dim) ))
    aug_images, aug_targets_xx, aug_targets_xy = np.zeros(images.shape), np.zeros(targets_xx.shape), np.zeros(targets_xy.shape)
    
    for i in range(len(indices)):
        
                 
        if i < n_roll:
 
            image = images[indices[i]].numpy()
            target_xx = targets_xx[indices[i]].numpy()
            target_xy = targets_xy[indices[i]].numpy()
            
            aug_images[i] = np.roll(images[indices[i]], trans[i], axis=[0,1] )
            aug_targets_xx[i] = np.roll( targets_xx[indices[i]], trans[i], axis=[0,1] )
            aug_targets_xy[i] =  np.roll( targets_xy[indices[i]], trans[i], axis=[0,1] )
                         
    aug_images[n_data_aug:] = images[indices[n_data_aug:] ]
    aug_targets_xx[n_data_aug:] = targets_xx[indices[n_data_aug:]]
    aug_targets_xy[n_data_aug:] = targets_xy[indices[n_data_aug:] ]
    
    if shuffle is False:
        aug_images = aug_images[np.argsort(indices)]
        aug_targets_xx = aug_targets_xx[np.argsort(indices)]
        aug_targets_xy = aug_targets_xy[np.argsort(indices)]
        
    return torch.from_numpy(aug_images), torch.from_numpy(aug_targets_xx), torch.from_numpy(aug_targets_xy)