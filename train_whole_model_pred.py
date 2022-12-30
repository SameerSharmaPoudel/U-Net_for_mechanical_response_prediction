import dataloader_xy
import torch
from torch.utils.tensorboard import SummaryWriter
from log_whole_model_prediction_data import log_whole_model_pred_image_data
from eval_metrics import compute_r_square
from model_utils import roll_data

def training(epoch, model, path, device, loss, optimizer):
    
    running_train_mse_loss = 0.0
    running_r_square_xx = 0.0
    running_r_square_xy = 0.0
    
    model.train() 
    
    for data in dataloader_xy.train_dataloader:

        image, target_xx, target_xy = data[0], data[1], data[2]
        
        if epoch%10 == 0:
            image, target_xx, target_xy = roll_data(image, target_xx, target_xy, part=0.50, shuffle=True)

        image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
        target = torch.cat([target_xx, target_xy], 1)                      

        encoded_image, x3, x5, x7 = model.encoder(image)
        decoded = model.decoder(encoded_image, x3, x5, x7)
        decoded_image_xx, decoded_image_xy = decoded[:,0,:,:],  decoded[:,1,:,:]
        decoded_image_xx, decoded_image_xy = torch.unsqueeze(decoded_image_xx, dim=1), torch.unsqueeze(decoded_image_xy, dim=1)
        
        loss_mse_xx = loss(target_xx, decoded_image_xx)
        loss_mse_xy = loss(target_xy, decoded_image_xy)
        loss_mse = 0.5*(loss_mse_xx + loss_mse_xy)
        loss_mse.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_train_mse_loss += loss_mse.item()
        
        r_square_xx =  compute_r_square(target_xx, decoded_image_xx)  
        r_square_xy =  compute_r_square(target_xy, decoded_image_xy)  
        running_r_square_xx += r_square_xx
        running_r_square_xy += r_square_xy

    training_r_square_xx =  running_r_square_xx/ len(dataloader_xy.train_dataloader) 
    training_r_square_xy =  running_r_square_xy/ len(dataloader_xy.train_dataloader) 
    training_r_square = [training_r_square_xx, training_r_square_xy ]
    training_loss = running_train_mse_loss / len(dataloader_xy.train_dataloader)
    print(f'Epoch:{epoch},  Whole Model Prediction Training Loss:{training_loss:.4f}')
    
    return training_r_square, training_loss, log_whole_model_pred_image_data(epoch,path,image,decoded,target,train=True)


def validation(epoch, model, path, device, loss, optimizer):
    
    with torch.no_grad():
    
        running_val_mse_loss = 0.0
        running_r_square_xx = 0.0
        running_r_square_xy = 0.0
        
        model.eval()
        
        for data in dataloader_xy.val_dataloader:              
        
            image, target_xx, target_xy = data[0], data[1], data[2]

            if epoch%10 == 0:
                image, target_xx, target_xy = roll_data(image, target_xx, target_xy, part=0.50, shuffle=True)

            image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
            target = torch.cat([target_xx, target_xy], 1)                           
            
            encoded_image, x3, x5, x7 = model.encoder(image)
            decoded = model.decoder(encoded_image, x3, x5, x7)
            decoded_image_xx, decoded_image_xy = decoded[:,0,:,:],  decoded[:,1,:,:]
            decoded_image_xx, decoded_image_xy = torch.unsqueeze(decoded_image_xx, dim=1), torch.unsqueeze(decoded_image_xy, dim=1)
                        
            loss_mse_xx = loss(target_xx, decoded_image_xx)
            loss_mse_xy = loss(target_xy, decoded_image_xy)
            loss_mse = 0.5*(loss_mse_xx + loss_mse_xy)
            running_val_mse_loss += loss_mse.item()
            
            r_square_xx =  compute_r_square(target_xx, decoded_image_xx)  
            r_square_xy =  compute_r_square(target_xy, decoded_image_xy)  
            running_r_square_xx += r_square_xx
            running_r_square_xy += r_square_xy
            
        validation_r_square_xx =  running_r_square_xx/ len(dataloader_xy.val_dataloader)                              
        validation_r_square_xy =  running_r_square_xy/ len(dataloader_xy.val_dataloader) 
        validation_loss = running_val_mse_loss / len(dataloader_xy.val_dataloader)
        validation_r_square = [validation_r_square_xx, validation_r_square_xy ]
        print(f'Epoch:{epoch}, Whole Model Prediction Validation Loss:{validation_loss:.4f}')
        
        return validation_r_square, validation_loss,  log_whole_model_pred_image_data(epoch,path,image,decoded,target,train=False)


def test(epoch, model, path, device, loss, optimizer):
    
    with torch.no_grad():
    
        running_test_mse_loss = 0.0
        running_r_square_xx = 0.0
        running_r_square_xy = 0.0
        model.eval()
        
        for data in dataloader_xy.test_dataloader:                
        
            image, target_xx, target_xy = data[0], data[1], data[2]
            
            image, target_xx, target_xy  = image.to(device), target_xx.to(device), target_xy.to(device)
            target = torch.cat([target_xx, target_xy], 1)
            
            encoded_image, x3, x5, x7 = model.encoder(image)
            decoded = model.decoder(encoded_image, x3, x5, x7)
            decoded_image_xx, decoded_image_xy = decoded[:,0,:,:],  decoded[:,1,:,:]
            decoded_image_xx, decoded_image_xy = torch.unsqueeze(decoded_image_xx, dim=1), torch.unsqueeze(decoded_image_xy, dim=1)
            
            loss_mse_xx = loss(target_xx, decoded_image_xx)
            loss_mse_xy = loss(target_xy, decoded_image_xy)
            loss_mse = 0.5*(loss_mse_xx + loss_mse_xy)
            running_test_mse_loss += loss_mse.item()
                        
            r_square_xx =  compute_r_square(target_xx, decoded_image_xx)  
            r_square_xy =  compute_r_square(target_xy, decoded_image_xy)  
            running_r_square_xx += r_square_xx
            running_r_square_xy += r_square_xy
            
        test_r_square_xx =  running_r_square_xx/ len(dataloader_xy.test_dataloader)                              
        test_r_square_xy =  running_r_square_xy/ len(dataloader_xy.test_dataloader) 
        test_loss = running_test_mse_loss / len(dataloader_xy.test_dataloader)
        test_r_square = 0.5 *(test_r_square_xx + test_r_square_xy )
        
        return test_r_square, test_loss