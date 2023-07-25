from cycle_gan import CycleGAN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import random
from collections import deque
from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/gan_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(config['data']['img_size'])])

# Set the data
img_refer = plt.imread(config['data']['img_refer_path'])
img_refer = transform(img_refer)
img_refer = img_refer.to(device)

img_style = plt.imread(config['data']['img_style_path'])
img_style = transform(img_style)
img_style = img_style.to(device)

# Set the model
model = CycleGAN(gen_input_dim=config['model']['gen_input_dim'], gen_output_dim=config['model']['gen_output_dim'],
                 dis_input_dim=config['model']['dis_input_dim'], dis_conv_filters=config['model']['dis_conv_filters'],
                 dis_conv_kernels=config['model']['dis_conv_kernels'], dis_conv_strides=config['model']['dis_conv_strides'],
                 dis_conv_pads=config['model']['dis_conv_pads'], dis_norm=config['model']['dis_norm']).to(device)

print(model, device)

# Set the criterion and optimizer
gxy_optimizer = optim.AdamW(model.Gxy.parameters(),
                          lr=config['train']['lr'],
                          betas=config['train']['betas'],
                          eps=config['train']['eps'],
                          weight_decay=config['train']['weight_decay'])
gyx_optimizer = optim.AdamW(model.Gyx.parameters(),
                          lr=config['train']['lr'],
                          betas=config['train']['betas'],
                          eps=config['train']['eps'],
                          weight_decay=config['train']['weight_decay'])
dx_optimizer = optim.AdamW(model.Dx.parameters(),
                          lr=config['train']['lr'],
                          betas=config['train']['betas'],
                          eps=config['train']['eps'],
                          weight_decay=config['train']['weight_decay'])
dy_optimizer = optim.AdamW(model.Dy.parameters(),
                          lr=config['train']['lr'],
                          betas=config['train']['betas'],
                          eps=config['train']['eps'],
                          weight_decay=config['train']['weight_decay'])
criterion = nn.BCELoss()

# Training
def train(epoch, model, realA, realB, gxy_optimizer, gyx_optimizer, dx_optimizer, dy_optimizer):
  model.train()
  
  fakeA_buffer = deque(maxlen=100)
  fakeB_buffer = deque(maxlen=100)
  batch_size = config['data']['batch_size']
  
  for i in range(epoch):
    fakeA = model.Gyx(realA)
    fakeB = model.Gxy(realB)
    
    fakeA_buffer.append(fakeA)
    fakeB_buffer.append(fakeB)
    
    fakeA_batch = random.sample(fakeA_buffer, 1)
    fakeB_batch = random.sample(fakeB_buffer, 1)
    
    realA_score = model.Dx(realA)
    fakeA_score = model.Dx(fakeA_batch)
    
    realB_score = model.Dy(realB)
    fakeB_score = model.Dy(fakeB_batch)
    
    
    
    
    # g_train_loss = 0.0
    # d_train_loss = 0.0
    # train_num = 0

    # # Transfer data to device
    # real_img = real_img.to(device)
    # real_score = model.D(real_img)
    # real_label = torch.ones(batch_size, 1, device=device)

    # # Generate generated image
    # z = 2 * torch.rand(batch_size, z_latent, device=device) - 1
    # fake_img = model.G(z)
    # fake_score = model.D(fake_img)
    # fake_label = torch.zeros(batch_size, 1, device=device)
    
    # # Loss for the discriminator
    # d_loss_real = criterion(real_score, real_label)
    # d_loss_fake = criterion(fake_score, fake_label)
    # d_loss = d_loss_real + d_loss_fake
    
    # # Training for the discriminator
    # d_optimizer.zero_grad()
    # d_loss.backward()
    # d_optimizer.step()
    
    # # Generator
    # # Get the fake images and scores
    # z = 2 * torch.rand(batch_size, z_latent, device=device) - 1
    # fake_img = model.G(z)
    # fake_score = model.D(fake_img)
    # real_label = torch.ones(batch_size, 1, device=device)
    
    # # Training for the generator
    # g_loss = criterion(fake_score, real_label)
    # g_optimizer.zero_grad()
    # g_loss.backward()
    # g_optimizer.step()

    # # loss
    # g_train_loss += g_loss.item()
    # d_train_loss += d_loss.item()
    # train_num += real_img.size(0)
    
  #   if i % config['others']['log_period'] == 0 and i != 0:
  #     print(f'[{epoch}, {i}]\t Train loss: (G){g_train_loss / train_num:.5f}, (D){d_train_loss / train_num:.5f}')
  
  # # Average loss
  # d_train_loss /= train_num
  
  return 0

# Main
if __name__ == '__main__':
  for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
    # Training
    train_loss = train(epoch, model, img_refer, img_style, gxy_optimizer, gyx_optimizer, dx_optimizer, dy_optimizer)
    
    # Print the log
    print(f'Epoch: {epoch}\t Train loss: {train_loss:.5f}\t')
    
    # Save the model
    save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=gxy_optimizer, loss=train_loss, config=config)
    