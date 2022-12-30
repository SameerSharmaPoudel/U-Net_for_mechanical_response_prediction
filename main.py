import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, lr_scheduler
from load_config_file import load_config
from model_utils import feed_model_to_device
from train_reconstruction import training_recon, validation_recon, test_recon
from train_decoder import training_decoder, validation_decoder, test_decoder
from train_whole_model_pred import training, validation, test
from log_reconstruction_data import log_recon_loss_and_Rsquare_score
from log_decoder_prediction_data import log_dec_pred_loss_and_Rsquare_score
from log_whole_model_prediction_data import log_whole_model_pred_loss_and_Rsquare_score
import csv
import time
import sys
import random
import os
sys.path.append('models/')
from models import model_list, model_name

###############################################################################
#                       Reconstruction Training
###############################################################################

config = load_config('config1.yaml')
parent_dir = 'new_data'

for model_idx in range(len(model_list)):
   
    print('Training the model:{}' .format(model_name[model_idx]))

    recon_start_time = time.time()
    
    directory = model_name[model_idx] + '_' + str(config['Hyperparameters']['train_batch_size']) + '_' + str(config['Hyperparameters']['lr']) + '_' + str(config['Data_log']['exp_no'])    
    path = os.path.join(parent_dir, directory)
    try:
        os.makedirs(path, exist_ok = True)
        print("Directory '%s' created successfully" %directory)
    except OSError as error:
        print("Directory '%s' can not be created")
    
    model = model_list[model_idx]
    model, device = feed_model_to_device(model, config)    
    rec_loss = MSELoss() 
    #rec_loss = L1Loss()    
    
    optimizer = Adam(model.parameters(), lr=config['Hyperparameters']['lr'], weight_decay=config['Hyperparameters']['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.01)
    
    #error and accuracy measures    
    recon_training_mse_loss = []
    recon_val_mse_loss = []    
    recon_training_r_square = []
    recon_validation_r_square = []
    
    stopping_delay = 400
    init_best_val_loss = 1e10
    
    for epoch in range(config['Hyperparameters']['num_epochs']):
            
        train_r_square, training_loss, _ = training_recon(epoch, model, path, device, rec_loss, optimizer)
        val_r_square, validation_loss, _  = validation_recon(epoch, model, path, device, rec_loss, optimizer)
        
        scheduler.step()
        
        recon_training_mse_loss.append(training_loss) 
        recon_val_mse_loss.append(validation_loss) 
        
        recon_training_r_square.append(train_r_square) 
        recon_validation_r_square.append(val_r_square) 
                   
        if validation_loss < init_best_val_loss :                      
            best_epoch = epoch
            val_loss_at_best_epoch  = validation_loss 
            stagnation = 0
            
            #####################SAVING THE RECONSTRUCTION MODEL##################
            checkpoint = {
                'epoch':best_epoch,
                'model_state': model.state_dict(),
                'optim_state':optimizer.state_dict(),
                }    
            PATH_recon_model = os.path.join(path, f'recon_model_{best_epoch}.pth')       
            torch.save(checkpoint, PATH_recon_model)
            ###########################################################################
            
        else:
            stagnation += 1
            
        if stagnation > stopping_delay:                  
            break
           
    print( 'Trained for {} epochs for reconstruction. Best epoch at {}'.format( epoch, best_epoch))   
    PATH_best_recon_model = os.path.join(path, f'recon_model_{best_epoch}.pth')
    print(f'Best recon model:{PATH_best_recon_model}')
    
    log_recon_loss_and_Rsquare_score(recon_training_mse_loss, recon_val_mse_loss, recon_training_r_square, recon_validation_r_square, path)

    best_recon_val_loss = min(recon_val_mse_loss)
    epoch_best_recon_val_loss = recon_val_mse_loss.index(best_recon_val_loss)
    
    recon_test_r_square, recon_test_loss = test_recon(epoch, model, path, device, rec_loss, optimizer)
    print(f'Reconstruction Test Loss:{recon_test_loss:.4f}, Reconstruction Test Accuracy:{recon_test_r_square:.4f}')
 
    recon_end_time = time.time()
    recon_total_time = recon_end_time - recon_start_time

    header = ['Best Epoch', 'Validation Loss At Best Epoch', 'Best Validation Loss', 'Epoch For Best Validation Loss', 'Test Loss', 'Test Accuracy', 'Training Time']
    data = [[best_epoch, val_loss_at_best_epoch, best_recon_val_loss, epoch_best_recon_val_loss, recon_test_loss, recon_test_r_square, recon_total_time ]]
    filename = os.path.join(path, 'recon_best_loss.csv')
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
              
    ###########################################################################
    #                          Loading Reconstruction Checkpoint
    ###########################################################################
    dec_start_time = time.time()
    
    model = model_list[model_idx]
    optimizer = Adam(model.parameters(), config['Hyperparameters']['lr'], weight_decay=config['Hyperparameters']['weight_decay'])
    
    PATH = PATH_best_recon_model 
                                                                     
    loaded_checkpoint = torch.load(PATH)
    epoch = loaded_checkpoint['epoch']
    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
    model, device = feed_model_to_device(model, config)  
    
    ###########################################################################
    #                          Decoder Training for Prediction
    ###########################################################################
    pred_loss = MSELoss() 
    #pred_loss = L1Loss()
    
    for param in model.encoder.parameters():
        param.requires_grad = False
  
    pred_optimizer = Adam(model.decoder.parameters(), config['Hyperparameters']['lr'], weight_decay=config['Hyperparameters']['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.01)
    
    dec_training_mse_loss = []
    dec_val_mse_loss = [] 
    
    dec_training_r_square_xx = []
    dec_validation_r_square_xx = []
    dec_training_r_square_xy = []
    dec_validation_r_square_xy = []
    
    init_best_val_loss = 1e10
    
    for epoch in range(config['Hyperparameters']['num_epochs']):
            
        train_r_square,training_loss, _  = training_decoder(epoch, model, path, device, pred_loss, pred_optimizer)
        val_r_square,validation_loss, _  = validation_decoder(epoch, model, path, device, pred_loss, pred_optimizer)
        
        scheduler.step()
        
        dec_training_mse_loss.append(training_loss) 
        dec_val_mse_loss.append(validation_loss) 
        
        dec_training_r_square_xx.append(train_r_square[0]) 
        dec_validation_r_square_xx.append(val_r_square[0]) 
        dec_training_r_square_xy.append(train_r_square[1]) 
        dec_validation_r_square_xy.append(val_r_square[1]) 
                
        if validation_loss < init_best_val_loss :                        
            best_epoch = epoch
            val_loss_at_best_epoch  = validation_loss
            stagnation = 0  
            
            ###############SAVING THE DECODER PREDICTION MODEL####################
            checkpoint = {
                'epoch':best_epoch,
                'model_state': model.state_dict(),
                'optim_state':optimizer.state_dict(),
                }    
            PATH_decoder = os.path.join(path, f'decoder_model_{best_epoch}.pth')       
            torch.save(checkpoint, PATH_decoder)
           ###############################################################################
        else:
            stagnation += 1
        
        if stagnation > stopping_delay:
            break
             
    print('Trained for {} epochs for prediction from decoder only. Best epoch at {}'.format( epoch, best_epoch))
    PATH_best_decoder = os.path.join(path, f'decoder_model_{best_epoch}.pth') 
    print(f'Best decoder:{PATH_best_decoder}')
     
    log_dec_pred_loss_and_Rsquare_score(dec_training_mse_loss, dec_val_mse_loss, dec_training_r_square_xx, dec_validation_r_square_xx, dec_training_r_square_xy, dec_validation_r_square_xy, path)
             
    best_dec_pred_val_loss = min(dec_val_mse_loss)
    epoch_best_dec_pred_val_loss = dec_val_mse_loss.index(best_dec_pred_val_loss)
    
    dec_test_r_square, dec_test_loss = test_decoder(epoch, model, path, device, pred_loss, optimizer)
    #print(dec_test_r_square, dec_test_loss)
    print(f'Decoder Test Loss:{dec_test_loss:.4f}, Decoder Test Accuracy:{dec_test_r_square:.4f}')
    
    dec_end_time = time.time()
    dec_total_time = dec_end_time - dec_start_time
    
    header = ['Best Epoch', 'Validation Loss At Best Epoch', 'Best Validation Loss', 'Epoch For Best Validation Loss', 'Test Loss', 'Test Accuracy', 'Training Time']
    data = [[best_epoch, val_loss_at_best_epoch, best_dec_pred_val_loss, epoch_best_dec_pred_val_loss, dec_test_loss, dec_test_r_square, dec_total_time ]]
    filename = os.path.join(path, 'dec_pred_best_loss.csv')
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    ###########################################################################
    #                            Loading Decoder Checkpoint
    ###########################################################################
    ae_start_time = time.time()
    
    model = model_list[model_idx]
    optimizer = Adam(model.parameters(), config['Hyperparameters']['lr'], weight_decay=config['Hyperparameters']['weight_decay'])
    
    PATH = PATH_best_decoder
    
    loaded_checkpoint = torch.load(PATH)
    epoch = loaded_checkpoint['epoch']
    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
    model, device = feed_model_to_device(model, config)  
    
    ###########################################################################
    #                         Whole Model Training for Prediction
    ###########################################################################
    pred_loss = MSELoss()
    #pred_loss = L1Loss()
    
    for param in model.encoder.parameters():
        param.requires_grad = True

    pred_optimizer = Adam(model.parameters(), config['Hyperparameters']['lr'], weight_decay=config['Hyperparameters']['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.01)
    
    training_mse_loss = []
    val_mse_loss = [] 
    
    training_r_square_xx = []
    validation_r_square_xx = []
    training_r_square_xy = []
    validation_r_square_xy = []
    
    init_best_val_loss = 1e10
    
    for epoch in range(config['Hyperparameters']['num_epochs']):
            
        train_r_square,training_loss, _  = training(epoch, model, path, device, pred_loss, pred_optimizer)
        val_r_square,validation_loss, _  = validation(epoch, model, path, device, pred_loss, pred_optimizer)
        
        scheduler.step()
        
        training_mse_loss.append(training_loss) 
        val_mse_loss.append(validation_loss) 
        
        training_r_square_xx.append(train_r_square[0]) 
        validation_r_square_xx.append(val_r_square[0]) 
        training_r_square_xy.append(train_r_square[1]) 
        validation_r_square_xy.append(val_r_square[1]) 
        
        if validation_loss < init_best_val_loss :                        
            best_epoch = epoch
            val_loss_at_best_epoch  = validation_loss 
            stagnation = 0 
            
            #####################SAVING THE WHOLE MODEL PREDICTION MODEL##########
            checkpoint = {
                'epoch':best_epoch,
                'model_state': model.state_dict(),
                'optim_state':optimizer.state_dict()
                } 
            PATH_whole_model_pred = os.path.join(path, f'whole_model_prediction_{best_epoch}.pth')  
            torch.save(checkpoint, PATH_whole_model_pred)
            ########################################################################### 
        else:
            stagnation += 1
        
        if stagnation > stopping_delay:  
            break
               
    print('Trained for {} epochs for prediction from whole model. Best epoch at {}'.format( epoch, best_epoch))
    PATH_best_whole_model_pred = os.path.join(path, f'whole_model_prediction_{best_epoch}.pth')
    print(f'Best AE for prediction:{PATH_best_whole_model_pred}')
       
    log_whole_model_pred_loss_and_Rsquare_score(training_mse_loss, val_mse_loss, training_r_square_xx, validation_r_square_xx, training_r_square_xy, validation_r_square_xy, path)
              
    best_whole_model_pred_val_loss = min(val_mse_loss)
    epoch_best_whole_model_pred_val_loss = val_mse_loss.index(best_whole_model_pred_val_loss)
    
    test_r_square, test_loss = test(epoch, model, path, device, pred_loss, optimizer)
    
    print(f'Whole Model Prediction Test Loss:{test_loss:.4f}, Whole Model Prediction Accuracy Loss:{test_r_square:.4f}')
    
    ae_end_time = time.time()
    ae_total_time = ae_end_time - ae_start_time
    
    header = ['Best Epoch', 'Validation Loss At Best Epoch', 'Best Validation Loss', 'Epoch For Best Validation Loss', 'Test Loss', 'Test_Accuracy', 'Training Time']
    data = [[best_epoch, val_loss_at_best_epoch, best_whole_model_pred_val_loss, epoch_best_whole_model_pred_val_loss, test_loss, test_r_square, ae_total_time]]
    filename = os.path.join(path, 'whole_model_pred_best_loss.csv')
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    total_time = recon_total_time + dec_total_time + ae_total_time
    filename = os.path.join(path, 'TimeLog.txt')
    f = open(filename, 'a+')
    content = [model_name[model_idx], total_time]
    f.writelines(str(content))
    f.close()   
    







