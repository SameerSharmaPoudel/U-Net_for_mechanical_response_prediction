import torch
from torch.nn import Linear, ReLU, LeakyReLU, Sigmoid, MaxUnpool2d, MSELoss, Sequential, Conv2d, Dropout2d, MaxPool2d, Module, Softmax, BatchNorm2d, ConvTranspose2d, AvgPool2d
from torch.optim import Adam
from torchinfo import summary
from torch.nn import functional as F

def up_conv(in_c, out_c):
    
    conv = Sequential(
        ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
        BatchNorm2d(out_c),
        LeakyReLU(0.01, inplace=True)    
        )
    return conv
#"""

class conv_block(Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = LeakyReLU(0.01, inplace= True)
        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3)),
        )

        self.branch3 = Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5)),
        )

        self.branch4 = Sequential(
            MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )


    def forward(self, x):
        return torch.cat(
            [self.branch1(x), 
             self.branch2(F.pad(x, pad=(1, 1, 1, 1), mode='circular')), 
             self.branch3(F.pad(x, pad=(2, 2, 2, 2), mode='circular')), 
             self.branch4(F.pad(x, pad=(1, 1, 1, 1), mode='circular'))], 1)
    
    
class Encoder(Module):
    def __init__(self,**kwargs):
        super().__init__()
        
        #Encoder
        self.down_conv_1 = AvgPool2d(2,2,0)
        
    def forward(self, x):
        
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_1(x2)
        encoded = x3.view(-1,x3.shape[2]*x3.shape[3],1,1)
        #print(encoded.shape)
        # compressed representation        
        return encoded, x1, x2

class Decoder(Module):
    def __init__(self,**kwargs):
        super().__init__()
            
        #Decoder
        #self.conv = Inception_block(1024, 256, 400, 512, 80, 128, 128)
        
        self.t_conv0 = up_conv(256, 256)
        #self.conv0 = Inception_block(256, 64, 128, 128, 32, 32, 32)
        self.t_conv1 = up_conv(256, 256)
        self.conv1 = Inception_block(256, 64, 128, 128, 32, 32, 32)
        self.t_conv2 = up_conv(256, 128)
        self.conv2 = Inception_block(128, 32, 64, 64, 16, 16, 16)
        self.t_conv3 = up_conv(128, 64)
        self.conv3 = Inception_block(64, 16, 32, 32, 8, 8, 8)
        self.t_conv4 = up_conv(64,32)
        self.conv4 = Inception_block(32, 8, 16, 16, 4, 4, 4)
        self.t_conv5 = up_conv(32,16)
        self.conv5 = Inception_block(16, 4, 8, 8, 2, 2, 2)
        self.t_conv6 = up_conv(16, 8)
        self.out = Conv2d(8, 2, 1)
        
        self.skip2 = Inception_block(1, 8, 16, 16, 4, 4, 4)
        self.skip1 = Inception_block(1, 4, 8, 8, 2, 2, 2)
        
    def forward(self, x, x1, x2):
                
        #Decoder

        x12 = self.t_conv0(x)
        #print(x12.shape)
        #x12 = self.conv0(x12)
        #print(x12.shape)
        x13 = self.t_conv1(x12)
        #print(x13.shape)
        x13 = self.conv1(x13)
        #print(x13.shape)
        x14 = self.t_conv2(x13)
        #print(x14.shape)
        #x14 = torch.cat([x14, x9],1)
        #print(x14.shape)
        x14 = self.conv2(x14)
        #print(x14.shape)
        x15 = self.t_conv3(x14)
        #print(x15.shape)
        #x15 = torch.cat([x15, x7],1)
        #print(x15.shape)
        x15 = self.conv3(x15)
        #print(x15.shape)
        x16 = self.t_conv4(x15)
        #print(x16.shape)
        x2 = self.skip2(x2)
        #x16 = torch.cat([x16, x2],1)
        #print(x16.shape)
        x16 = self.conv4(x16)        
        #print(x16.shape)
        x17 = self.t_conv5(x16)
        #print(x17.shape)
        x1 = self.skip1(x1)
        #x17 = torch.cat([x17, x1],1)
        #print(x17.shape)
        x17 = self.conv5(x17)
        #print(x17.shape)
        x18 = self.t_conv6(x17)
        #print(x18.shape)
        decoded = self.out(x18)
        #print(decoded.shape) 
        return decoded
    
class AE(Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x,x1,x2 = self.encoder(x)
        x = self.decoder(x,x1,x2)

        return x

model = AE().double()


#"""
if __name__ == "__main__":
    
    image = torch.rand((32,1,128,128))
    model = AE()
    print(model(image))
    print(summary(model, (32,1,128,128)))
#_""" 
