import torch
import dataloader_xy
from torch.utils.tensorboard import SummaryWriter
from eval_metrics import compute_r_square
from log_reconstruction_data import log_recon_image_data
from model_utils import roll_data

def training_recon(epoch, model, path, device, loss, optimizer):
       
    running_train_mse_loss = 0.0
    running_r_square = 0.0
    model.train() 
    
    for data in dataloader_xy.train_dataloader:                

        image, target_xx, target_xy = data[0], data[1], data[2]
        
        if epoch%10 == 0:
            image, _, _ = roll_data(image, target_xx, target_xy, part=0.50, shuffle=True) 
            
        image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
        
        encoded_image, x3, x5, x7 = model.encoder(image)
        reconstructed_image = model.decoder(encoded_image, x3, x5, x7)
        
        reconstructed_image = reconstructed_image[:,0,:,:] 
        reconstructed_image = torch.unsqueeze(reconstructed_image, dim=1)
        loss_mse = loss(image,reconstructed_image) 
        running_train_mse_loss += loss_mse.item() 
        loss_mse.backward()       
        optimizer.step()
        optimizer.zero_grad()
                
        r_square =  compute_r_square(image, reconstructed_image)  
        running_r_square += r_square   
    
    training_r_square =  running_r_square/ len(dataloader_xy.train_dataloader)   
    training_loss = running_train_mse_loss / len(dataloader_xy.train_dataloader)
    print(f'Epoch:{epoch}, Reconstruction Training Loss:{training_loss:.4f}')    
    return training_r_square, training_loss, log_recon_image_data(epoch,path,image,encoded_image,reconstructed_image,train=True)


def validation_recon(epoch, model, path, device, loss, optimizer):
    
    with torch.no_grad():
        
        running_val_mse_loss = 0.0
        running_r_square = 0.0
        model.eval()
        
        for data in dataloader_xy.val_dataloader:                
        
            image, target_xx, target_xy = data[0], data[1], data[2]
            
            if epoch%10 == 0:
                image, _, _ = roll_data(image, target_xx, target_xy, part=0.50, shuffle=True)   
            
            image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
            
            encoded_image, x3, x5, x7 = model.encoder(image)
            reconstructed_image = model.decoder(encoded_image, x3, x5, x7)

            reconstructed_image = reconstructed_image[:,0,:,:] 
            reconstructed_image = torch.unsqueeze(reconstructed_image, dim=1)
            loss_mse = loss(image,reconstructed_image) 
            running_val_mse_loss += loss_mse.item()
            
            r_square =  compute_r_square(image, reconstructed_image)
            running_r_square += r_square
            
        validation_r_square =  running_r_square/len(dataloader_xy.val_dataloader)            
        validation_loss = running_val_mse_loss / len(dataloader_xy.val_dataloader)
        print(f'Epoch:{epoch}, Reconstruction Validation Loss:{validation_loss:.4f}')
     
        return validation_r_square, validation_loss, log_recon_image_data(epoch,path,image,encoded_image,reconstructed_image, train=False)


def test_recon(epoch, model, path, device, loss, optimizer):
    
    with torch.no_grad():
        
        running_test_mse_loss = 0.0
        running_r_square = 0.0
        model.eval()
        
        for data in dataloader_xy.test_dataloader:                
        
            image, target_xx, target_xy = data[0], data[1], data[2]
             
            image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
            
            encoded_image, x3, x5, x7 = model.encoder(image)
            reconstructed_image = model.decoder(encoded_image, x3, x5, x7)

            reconstructed_image = reconstructed_image[:,0,:,:] 
            reconstructed_image = torch.unsqueeze(reconstructed_image, dim=1)
            loss_mse = loss(image,reconstructed_image) 
            running_test_mse_loss += loss_mse.item()
            
            r_square =  compute_r_square(image, reconstructed_image)
            running_r_square += r_square
            
        test_r_square =  running_r_square/len(dataloader_xy.test_dataloader)            
        test_loss = running_test_mse_loss / len(dataloader_xy.test_dataloader)
        
        return test_r_square, test_loss
