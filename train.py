import torch, model
import torch.nn as nn
import torch.optim as optim


class DCGAN():
    def __init__(self, z_dim=100, gpu_id=0, print_freq=10, epoch_print=10):
        self.z_dim = z_dim
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.BCELoss().cuda(self.gpu)
        
        self.G = model.Generator(self.z_dim).cuda(self.gpu)
        self.D = model.Discriminator().cuda(self.gpu)

        self.train_G_losses = []
        self.train_D_losses = []

    def train(self, dataloader, epochs, lr, beta1):
        self.G.train()
        self.D.train()
        optimizer_G = optim.Adam(self.G.parameters(), lr, betas=(beta1, 0.999))
        optimizer_D = optim.Adam(self.D.parameters(), lr, betas=(beta1, 0.999))
        
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0: print('Epoch {} Started...'.format(epoch+1))
            for i, (X, _) in enumerate(dataloader):
                n = X.size(0)
                real = torch.ones(n).cuda(self.gpu)
                fake = torch.zeros(n).cuda(self.gpu)
                
                # Discriminator
                self.D.zero_grad()
                
                X = X.cuda(self.gpu)
                real_loss = self.loss_function(self.D(X), real)
                real_loss.backward()
                
                noise = torch.randn(n, self.z_dim, 1, 1).cuda(self.gpu)
                fake_images = self.G(noise)
                fake_loss = self.loss_function(self.D(fake_images.detach()), fake)
                fake_loss.backward()

                optimizer_D.step()
                D_loss = real_loss.item() + fake_loss.item()
                
                # Generator
                self.G.zero_grad()
                loss_G = self.loss_function(self.D(fake_images), real)
                loss_G.backward()
                
                optimizer_G.step()
                G_loss = loss_G.item()

                if (i+1) % self.print_freq == 0:
                    self.train_G_losses.append(G_loss)
                    self.train_D_losses.append(D_loss)

                    if epoch % self.epoch_print == 0:
                        state = ('Iteration : {} - G Loss : {:.6f}, D Loss : {:.6f}').format(i+1, G_loss, D_loss)
                        print(state)