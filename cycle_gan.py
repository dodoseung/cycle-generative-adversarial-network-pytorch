import torch
import torch.nn.functional as F
import torch.nn as nn

# Cycle Ganerative Adversarial Network
class CycleGAN(nn.Module):
    def __init__(self, gen_input_dim=3, gen_output_dim=3,
                 dis_input_dim=3, dis_conv_filters=[16,32,64,128], dis_conv_kernels=[3,3,3,3],
                 dis_conv_strides=[2,2,2,1], dis_conv_pads=[1,1,1,1], dis_norm=[False, True, True, True]):
        super(CycleGAN, self).__init__()
        
        # Set the generator and discriminator
        self.Gxy = UNet(gen_input_dim, gen_output_dim)
        self.Dy = Discriminator(dis_input_dim, dis_conv_filters, dis_conv_kernels, dis_conv_strides, dis_conv_pads, dis_norm)
        self.Gyx = UNet(gen_input_dim, gen_output_dim)
        self.Dx = Discriminator(dis_input_dim, dis_conv_filters, dis_conv_kernels, dis_conv_strides, dis_conv_pads, dis_norm)


class Discriminator(nn.Module):
    def __init__(self, input_dim, conv_filters, conv_kernels, conv_strides, conv_pads, norm):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads
        self.norm = norm
        
        # Append filters for the init layer
        self.num_layers = len(self.conv_filters)
        self.conv_filters.insert(0, self.input_dim)
        
        # Convolution layers
        self.conv = []
        for i in range(self.num_layers):
            if self.norm:
                layer = nn.Sequential(
                    nn.Conv2d(self.conv_filters[i], self.conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding=self.conv_pads[i]),
                    nn.InstanceNorm2d(conv_filters[i+1]),
                    nn.LeakyReLU(negative_slope=0.2))
            else:
                layer = nn.Sequential(
                    nn.Conv2d(self.conv_filters[i], self.conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding=self.conv_pads[i]),
                    nn.InstanceNorm2d(conv_filters[i+1]))
                
            self.conv.append(layer)
        self.conv = nn.ModuleList(self.conv)
        
        # Output layer
        self.out = nn.Conv2d(self.conv_filters[-1], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Delete the item
        del self.conv_filters[0]
        
    def forward(self, x):
        # CNN
        for i in range(self.num_layers):
            x = self.conv[i](x)
        
        # Output
        x = self.out(x)
        
        return x

# U-Net for the generator
class UNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(UNet, self).__init__()
        # Initiation
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Batch norm
        self.sample_norm32 = nn.InstanceNorm2d(32)
        self.sample_norm64 = nn.InstanceNorm2d(64)
        self.sample_norm128 = nn.InstanceNorm2d(128)
        self.sample_norm256 = nn.InstanceNorm2d(256)

        
        # Encoder (Contracting path)
        self.enc_conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        # Decoder (Expanding path)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.out = nn.Conv2d(32, self.output_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
    def forward(self, x):
        # Encoding
        x = F.relu(self.sample_norm32(self.enc_conv1(x)))
        res1 = x
        x = F.relu(self.sample_norm64(self.enc_conv2(x)))
        res2 = x
        x = F.relu(self.sample_norm128(self.enc_conv3(x)))
        res3 = x
        x = F.relu(self.sample_norm256(self.enc_conv4(x)))

        # Decoding
        x = self.upsample(x)
        x = F.relu(self.sample_norm128(self.dec_conv1(x)))
        x = x + res3
        
        x = self.upsample(x)
        x = F.relu(self.sample_norm64(self.dec_conv2(x)))
        x = x + res2
        
        x = self.upsample(x)
        x = F.relu(self.sample_norm64(self.dec_conv3(x)))
        x = x + res1

        x = self.upsample(x)
        x = F.tanh(self.out(x))
        
        return x