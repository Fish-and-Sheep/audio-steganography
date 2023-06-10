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


class chat_model(nn.Module):
    def __init__(self, mid_size = 100):
        super(chat_model, self ).__init__() 
        self.chat_linear = nn.Sequential(
            nn.Linear(3,mid_size),
            nn.Linear(mid_size,mid_size),
            nn.Linear(mid_size,1),
            nn.Sigmoid()
        )

    def forward(self, mid_feature):
        batch_size = mid_feature.shape[0]
        channel_size = mid_feature.shape[1]
        feature_size = mid_feature.shape[2]
        mid_feature = mid_feature.view(batch_size*channel_size, feature_size)
        avg_f = torch.mean(mid_feature, dim=1).unsqueeze(0)
        max_f = torch.max(mid_feature, dim=1).values.unsqueeze(0)
        std_f = torch.std(mid_feature, dim=1).unsqueeze(0)
        w = self.chat_linear(torch.cat([avg_f,std_f,max_f],dim=0).T)
        mid_feature = mid_feature*w
        mid_feature = mid_feature.view(batch_size,channel_size,feature_size)
        return mid_feature


class redundant_encoder(nn.Module):
    def _name(self):
        return "redundant_encoder"

    def _conv1d(self, in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        #1×1
        #self.up1 = nn.Sequential(
            #nn.Upsample(scale_factor=512, mode='bilinear')
        #)

        #16×16
        #self.up1 = nn.Sequential(
            #nn.Upsample(scale_factor=32, mode='bilinear')
        #)

        # 10×10
        #self.up1 = nn.Sequential(
            #nn.Upsample(scale_factor=50, mode='bilinear')
        #)

        # 512×512
        #self.up1 = nn.Sequential(
            #nn.Upsample(scale_factor=1, mode='bilinear')
        #)

        # 256×256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        

        self.conv = nn.Sequential(
            self._conv1d(1, self.data_depth)
        )
        self.conv_expand = nn.Sequential(self._conv1d(1, self.data_depth))
        self.conv_img = convblock(self.channels_size, self.hidden_size)
        self.conv1 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv2 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv3 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv4 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv5 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv6 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv7 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv8 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv9 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv10 = convblock(self.hidden_size + self.data_depth, self.hidden_size)
        self.conv_out = nn.Sequential(self._conv1d(self.hidden_size, self.channels_size))

    def __init__(self, data_depth, hidden_size, channels_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._build_models()
        self.name = self._name()
    
    def forward(self, cover, data, factor=1):
        data = self.up1(data)
        N, C, H, W = data.size()
        data = self.conv_expand(data.view(N,C,H*W))
        

        x = self.conv_img(cover)
        x_1 = self.conv1(torch.cat([x, data], dim=1))
        x_2 = self.conv2(torch.cat([x_1, data], dim=1))
        x_3 = self.conv3(torch.cat([x_2, data], dim=1))
        x_4 = self.conv4(torch.cat([x_3, data], dim=1))
        x_5 = self.conv5(torch.cat([x_4, data], dim=1))
        x_6 = self.conv6(torch.cat([x_5, data], dim=1))
        x_7 = self.conv7(torch.cat([x_6, data], dim=1))
        x_8 = self.conv8(torch.cat([x_7, data], dim=1))
        x_9 = self.conv9(torch.cat([x_8, data], dim=1))
        x_10 = self.conv10(torch.cat([x_9, data], dim=1))
        x_out = self.conv_out(torch.cat([x_10], dim=1))
        x_out=cover + x_out*factor

        return x_out
