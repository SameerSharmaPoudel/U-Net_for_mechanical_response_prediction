import numpy as np
import torch

def compute_error_metrics(target, prediction):
    
    """
    Compute the error between predicted values (prediction) and ground truth values (target).

    INPUT(s):
        target      : torch.tensor of the ground truth values, dimension: batch_size * no_of_channels * height * width
        prediction     : torch.tensor of the predicted values from model, dimension: batch_size * no_of_channels * height * width

    OUTPUT(s):
        mae         : Mean Absolute Error (scalar value)
        mse         : Mean Squared Error (scalar value)

    """
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    for i in range(len(prediction)):  #loops through the length of batchsize
               
        mae =  np.abs(prediction[i,:,:] - target[i,:,:])
        mae += mae
        
        mse = (prediction[i,:,:] - target[i,:,:])**2
        mse += mse
               
    mae = np.mean(mae / len(prediction))
    mse = np.mean(mse / len(prediction))

    return mae, mse

def compute_r_square(target, prediction):
    
    """
    Compute the r-squared value to evaluate the goodness of fit of the regression model
    
    Formula: (ref:https://en.wikipedia.org/wiki/Coefficient_of_determination) 
    R2 = 1 - ( SSE / SST)
        where SSE = sum squared error between target and prediction 
              SST = sum squared error between target and mean of target
              
    returns scalar value of the coefficient of determination (R^2)
    """

    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    for i in range(len(prediction)):  #loops through the length of batchsize
        
        SSE = 0.0
        SST = 0.0
        
        SSE = np.sum((prediction[i,:,:] - target[i,:,:])**2)
        SSE += SSE
        
        SST = np.sum((target[i,:,:] - target[i,:,:].mean())**2)
        SST += SST

    r_square_score = 1 - SSE/SST
    
    return r_square_score