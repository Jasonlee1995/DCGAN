import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False), 
                                       nn.BatchNorm2d(1024), nn.ReLU(True),
                                       
                                       nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False), 
                                       nn.BatchNorm2d(512), nn.ReLU(True),
                                       
                                       nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), 
                                       nn.BatchNorm2d(256), nn.ReLU(True),
                                       
                                       nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), 
                                       nn.BatchNorm2d(128), nn.ReLU(True),
                                       
                                       nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False), 
                                       nn.Tanh())
        self._initialize_weights()

    def forward(self, x):
        return self.generator(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False), 
                                           nn.LeakyReLU(0.2, inplace=True),
                                           
                                           nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), 
                                           nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                                           
                                           nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), 
                                           nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                                           
                                           nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), 
                                           nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                                           
                                           nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False), 
                                           nn.Sigmoid())
        self._initialize_weights()

    def forward(self, x):
        output = self.discriminator(x)
        output = output.view(-1)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)