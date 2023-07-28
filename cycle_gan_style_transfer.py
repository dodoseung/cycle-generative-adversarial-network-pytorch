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
import copy

# Set the configuration
config = load_yaml("./config/cycle_gan_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(config['data']['img_size'])])

# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# apple2orange

# Set the training data
data = datasets.ImageFolder(root=config['data']['data_path'], transform=transform)
test_dataA = copy.deepcopy(data)
test_dataA.targets = test_dataA.targets[test_dataA.targets == 0]
test_dataA.imgs = test_dataA.imgs[test_dataA.targets == 0]

test_dataB = copy.deepcopy(data)
test_dataB.targets = test_dataB.targets[test_dataB.targets == 0]
test_dataB.imgs = test_dataB.imgs[test_dataB.targets == 0]

train_dataA = copy.deepcopy(data)
train_dataA.targets = train_dataA.targets[train_dataA.targets == 0]
train_dataA.imgs = train_dataA.imgs[train_dataA.targets == 0]

train_dataB = copy.deepcopy(data)
train_dataB.targets = train_dataB.targets[train_dataB.targets == 0]
train_dataB.imgs = train_dataB.imgs[train_dataB.targets == 0]

train_loaderA = torch.utils.data.DataLoader(train_dataA, batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

train_loaderB = torch.utils.data.DataLoader(train_dataB, batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

test_loaderA = torch.utils.data.DataLoader(test_dataA, batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

test_loaderB = torch.utils.data.DataLoader(test_dataB, batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

# Set the model
model = CycleGAN(gen_input_dim=config['model']['gen_input_dim'], gen_output_dim=config['model']['gen_output_dim'],
                 dis_input_dim=config['model']['dis_input_dim'], dis_conv_filters=config['model']['dis_conv_filters'],
                 dis_conv_kernels=config['model']['dis_conv_kernels'], dis_conv_strides=config['model']['dis_conv_strides'],
                 dis_conv_pads=config['model']['dis_conv_pads'], dis_norm=config['model']['dis_norm']).to(device)

print(model, device)

# Set the criterion and optimizer
gxy_optimizer = optim.Adam(model.Gxy.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'])
gyx_optimizer = optim.Adam(model.Gyx.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'])
dx_optimizer = optim.Adam(model.Dx.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'])
dy_optimizer = optim.Adam(model.Dy.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'])
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

fakeA_buffer = deque(maxlen=100)
fakeB_buffer = deque(maxlen=100)

# Training
def train(epoch, model, train_loaderA, train_loaderB, gxy_optimizer, gyx_optimizer, dx_optimizer, dy_optimizer):
  model.train()
  
  batch_size = config['data']['batch_size']
  valid = torch.ones(batch_size, 1, 16, 16, device=device)
  fake = torch.ones(batch_size, 1, 16, 16, device=device)
  
  losses = [0, 0, 0, 0]
  i = 0
  
  for dataA, dataB in zip(train_loaderA, train_loaderB):
    # Data
    realA, _ = dataA
    realA = realA.to(device)
    
    realB, _ = dataB
    realB = realB.to(device)

    fakeA = model.Gyx(realB)
    fakeB = model.Gxy(realA)
    
    # fakeA_buffer.append(fakeA)
    # fakeB_buffer.append(fakeB)
    
    # fakeA = fakeA_buffer[random.randrange(len(fakeA_buffer))]
    # fakeB = fakeB_buffer[random.randrange(len(fakeB_buffer))]
    
    realA_score = model.Dx(realA)
    fakeA_score = model.Dx(fakeA)

    realB_score = model.Dy(realB)
    fakeB_score = model.Dy(fakeB)
    
    # Discriminator Dx (Validity)
    # (torch.mean((realA_score - 1)**2) + torch.mean(fakeA_score**2))
    realA_loss = mse_loss(realA_score, valid)
    fakeA_loss = mse_loss(fakeA_score, fake)
    dx_loss = (realA_loss + fakeA_loss).mean()
    dx_optimizer.zero_grad()
    dx_loss.backward()
    dx_optimizer.step()
    
    # Discriminator Dy (Validity)
    realB_loss = mse_loss(realB_score, valid)
    fakeB_loss = mse_loss(fakeB_score, fake)
    dy_loss = (realB_loss + fakeB_loss).mean()
    dy_optimizer.zero_grad()
    dy_loss.backward()
    dy_optimizer.step()
    
    # Generator Gxy (Reconstruction and identity)
    scoreB = model.Dy(model.Gxy(realA))
    reconB = model.Gxy(model.Gyx(realB))
    idenB = model.Gxy(realB)
    gxy_loss = mse_loss(scoreB, valid) + l1_loss(realB, reconB) + l1_loss(realB, idenB)
    gxy_optimizer.zero_grad()
    gxy_loss.backward()
    gxy_optimizer.step()
    
    # Generator Gyx (Reconstruction and identity)
    scoreA = model.Dx(model.Gyx(realB))
    reconA = model.Gyx(model.Gxy(realA))
    idenA = model.Gyx(realA)
    gyx_loss = mse_loss(scoreA, valid) + l1_loss(realA, reconA) + l1_loss(realA, idenA)
    gyx_optimizer.zero_grad()
    gyx_loss.backward()
    gyx_optimizer.step()
    
    losses[0] += dx_loss
    losses[1] += dy_loss
    losses[2] += gxy_loss
    losses[3] += gyx_loss
    i = i + 1
    if i % config['others']['log_period'] == 0 and i != 0:
      num_data = i * batch_size
      print(f'[{epoch}, {i}]\t dx loss: {losses[0]/num_data:.5f}\t dy loss: {losses[1]/num_data:.5f}\t gxy loss: {losses[2]/num_data:.5f}\t gyx loss: {losses[3]/num_data:.5f}')
      losses = [0, 0, 0, 0]
  
  return dx_loss, dy_loss, gxy_loss, gyx_loss

# Main
if __name__ == '__main__':
  for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
    # Training
    dx_loss, dy_loss, gxy_loss, gyx_loss = train(epoch, model, train_loaderA, train_loaderB, gxy_optimizer, gyx_optimizer, dx_optimizer, dy_optimizer)
    
    # Print the log
    print(f'Epoch: {epoch}\t dx loss: {dx_loss:.5f}\t dy loss: {dy_loss:.5f}\t gxy loss: {gxy_loss:.5f}\t gyx loss: {gyx_loss:.5f}')
    
    # Save the model
    save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=gxy_optimizer, loss=gxy_loss, config=config)
    