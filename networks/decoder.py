from math import sqrt
import torch
from torch import nn
from utils.hparameter import Instance_Norm
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def convblock(in_channels, out_channels):
    def conv1d(in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(in_channels, out_channels):
        if Instance_Norm:
            return nn.Sequential(
            conv1d(in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm1d(out_channels, affine=True),
            )
        return nn.Sequential(
            conv1d(in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    return forward(in_channels, out_channels)


class redundant_decoder(nn.Module):
    def _name(self):
        return "redundant_decoder"

    def _build_models(self):
        self.conv1 = convblock(self.channels_size, self.hidden_size)
        self.conv2 = convblock(self.hidden_size, self.hidden_size)
        self.conv3 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv4 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv5 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv6 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv7 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv8 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv9 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv10 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv11 = convblock(self.hidden_size*2, self.hidden_size)

        self.conv_out = nn.Sequential(
            # nn.Conv2d(self.hidden_size, 1, 12, 8, 2),
            # 256×256
            nn.Conv2d(self.hidden_size, 1, kernel_size=15, stride=2, padding=7),
            # 512×512
            #nn.Conv2d(self.hidden_size, 1, kernel_size=15, stride=1, padding=7),
            # 10×10
            #nn.Conv2d(self.hidden_size, 1, kernel_size=15, stride=10, padding=3),
            # 16×16
            #nn.Conv2d(self.hidden_size, 1, kernel_size=15, stride=8, padding=8),
            # 1×1
            #nn.Conv2d(self.hidden_size, 1, kernel_size=15, stride=32, padding=8),
            # nn.Sigmoid(),
        )

        self.conv_out2 = nn.Sequential(
            # 512×512 256×256
            nn.Conv2d(1, 1, kernel_size=9, stride=1, padding=4),
            # 16×16
            #nn.Conv2d(1, 1, kernel_size=9, stride=4, padding=3),
            # 10×10
            #nn.Conv2d(1, 1, kernel_size=9, stride=5, padding=2),
            # 1×1
            #nn.Conv2d(1, 1, kernel_size=9, stride=32, padding=3),
            # nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d(output_size=(payload_size,payload_size)),
        )
 

      


    def __init__(self, data_depth, hidden_size, channels_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._build_models()
        self.name = self._name()

    def forward(self, image):
        x = self.conv1(image)
        x_list = [x]
        x_1 = self.conv2(torch.cat(x_list, dim=1))
        x_list.append(x_1)
        x_2 = self.conv3(torch.cat(x_list, dim=1))
        x_list = [x, x_2]
        x_3 = self.conv4(torch.cat(x_list, dim=1))
        x_list = [x, x_3]
        x_4 = self.conv5(torch.cat(x_list, dim=1))
        x_list = [x, x_4]
        x_5 = self.conv6(torch.cat(x_list, dim=1))
        x_list = [x, x_5]
        x_6 = self.conv7(torch.cat(x_list, dim=1))
        x_list = [x, x_6]
        x_7 = self.conv8(torch.cat(x_list, dim=1))
        x_list = [x, x_7]
        x_8 = self.conv9(torch.cat(x_list, dim=1))
        x_list = [x, x_8]
        x_9 = self.conv10(torch.cat(x_list, dim=1))
        x_list = [x, x_9]
        x_10 = self.conv11(torch.cat(x_list, dim=1))

        N, C, L = x_10.size()
        H = int(sqrt(L))
        x_10 = x_10.view(N,C,H,H)
        x_out = self.conv_out(x_10)
        x_out = self.conv_out2(x_out)
        return x_out
