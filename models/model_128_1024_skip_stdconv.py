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
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, **kwargs)
        self.batchnorm = BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
"""
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
"""    
    
class Encoder(Module):
    def __init__(self,**kwargs):
        super().__init__()
        
        #Encoder
        self.down_conv_1 =  conv_block(1,8)
        self.down_conv_2 =  conv_block(8,16)
        self.down_conv_3 =  conv_block(16,32)
        self.down_conv_4 =  conv_block(32,64)
        self.down_conv_5 =  conv_block(64,128)
        self.down_conv_6 =  conv_block(128,256)
        self.pool = MaxPool2d(2,2,0)
        
    def forward(self, x):
        
        #Encoder
        x1 = F.pad(x, pad=(1, 1, 1, 1), mode='circular')
        x1 = self.down_conv_1(x1)
        #print(x1.shape)
        x2 = self.pool(x1)
        #print(x2.shape)
        x3 = F.pad(x2, pad=(1, 1, 1, 1), mode='circular')
        x3 = self.down_conv_2(x3)
        #print(x3.shape)
        x4 = self.pool(x3)
        #print(x4.shape)
        x5 = F.pad(x4, pad=(1, 1, 1, 1), mode='circular')
        x5 = self.down_conv_3(x5)
        #print(x5.shape)
        x6 = self.pool(x5)
        #print(x6.shape)
        x7 = F.pad(x6, pad=(1, 1, 1, 1), mode='circular')
        x7 = self.down_conv_4(x7)
        #print(x7.shape)
        x8 = self.pool(x7)
        #print(x8.shape)
        x9 = F.pad(x8, pad=(1, 1, 1, 1), mode='circular')
        x9 = self.down_conv_5(x9)
        #print(x9.shape)
        x10 = self.pool(x9)
        #print(x10.shape)
        x11 = F.pad(x10, pad=(1, 1, 1, 1), mode='circular')
        x11 = self.down_conv_6(x11)
        #print(x11.shape)
        x11 = self.pool(x11)
        #print(x11.shape)
        encoded = x11.view(-1,x11.shape[1]*x11.shape[2]*x11.shape[3],1,1)
        #print(encoded.shape)
        # compressed representation        
        return encoded,x3,x5,x7

class Decoder(Module):
    def __init__(self,**kwargs):
        super().__init__()
            
        #Decoder
        self.latent_conv = Conv2d(1024,1024,1)
        """
        self.conv = Inception_block(256, 64, 128, 128, 32, 32, 32)
        self.t_conv1 = up_conv(256, 256)
        self.conv1 = Inception_block(256, 64, 128, 128, 32, 32, 32)
        self.t_conv2 = up_conv(256, 128)
        self.conv2 = Inception_block(128, 32, 64, 64, 16, 16, 16)
        self.t_conv3 = up_conv(128, 64)
        self.conv3 = Inception_block(128, 16, 32, 32, 8, 8, 8)
        self.t_conv4 = up_conv(64,32)
        self.conv4 = Inception_block(64, 8, 16, 16, 4, 4, 4)
        self.t_conv5 = up_conv(32,16)
        self.conv5 = Inception_block(32, 4, 8, 8, 2, 2, 2)
        self.t_conv6 = up_conv(16, 8)
        self.out = Conv2d(8, 2, 1)
        """
        self.t_conv0 = up_conv(1024, 256)
        self.conv0 =  conv_block(256,256)
        self.t_conv1 = up_conv(256, 256)
        self.conv1 =  conv_block(256,256)
        self.t_conv2 = up_conv(256, 128)
        self.conv2 =  conv_block(128,128)
        self.t_conv3 = up_conv(128, 64)
        self.conv3 =  conv_block(128,64)
        self.t_conv4 = up_conv(64,32)
        self.conv4 =  conv_block(64,32)
        self.t_conv5 = up_conv(32,16)
        self.conv5 =  conv_block(32,16)
        self.t_conv6 = up_conv(16, 8)
        self.out = Conv2d(8, 2, 1)
        
    def forward(self, x,x3,x5,x7):
                
        #Decoder
        x12 = self.latent_conv(x)
        x12 = self.t_conv0(x12)
        x12 = F.pad(x12, pad=(1, 1, 1, 1), mode='circular')
        x12 = self.conv0(x12)
        #print(x12.shape)
        x13 = self.t_conv1(x12)
        #print(x13.shape)
        x13 = F.pad(x13, pad=(1, 1, 1, 1), mode='circular')
        x13 = self.conv1(x13)
        #print(x13.shape)
        x14 = self.t_conv2(x13)
        #print(x14.shape)
        #x14 = torch.cat([x14, x9],1)
        #print(x14.shape)
        x14 = F.pad(x14, pad=(1, 1, 1, 1), mode='circular')
        x14 = self.conv2(x14)
        #print(x14.shape)
        x15 = self.t_conv3(x14)
        #print(x15.shape)
        x15 = torch.cat([x15, x7],1)
        #print(x15.shape)
        x15  = F.pad(x15, pad=(1, 1, 1, 1), mode='circular')
        x15 = self.conv3(x15)
        #print(x15.shape)
        x16 = self.t_conv4(x15)
        #print(x16.shape)
        x16 = torch.cat([x16, x5],1)
        #print(x16.shape)
        x16 = F.pad(x16, pad=(1, 1, 1, 1), mode='circular')
        x16 = self.conv4(x16)        
        #print(x16.shape)
        x17 = self.t_conv5(x16)
        #print(x17.shape)
        x17 = torch.cat([x17, x3],1)
        #print(x17.shape)
        x17 = F.pad(x17, pad=(1, 1, 1, 1), mode='circular')
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
        x,x3,x5,x7 = self.encoder(x)
        x = self.decoder(x,x3,x5,x7)

        return x

model = AE().double()


#"""
if __name__ == "__main__":
    
    image = torch.rand((32,1,128,128))
    model = AE()
    print(model(image))
    print(summary(model, (32,1,128,128)))
#_""" 