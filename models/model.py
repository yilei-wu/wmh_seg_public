# Author: Yilei Wu Email: yileiwu@outlook.com
# implementation refer https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class SE_block(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_Block, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1 )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1 )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1 )
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.se = SE_block(out_channels)

    def forward(self, x):
        identity = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(identity)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += identity
        out = F.leaky_relu(out)

        return out

class Model(nn.Module):
  def __init__(self, pre_trained = False, decoder = None):
    super(Model, self).__init__()

    self.bottleneck = []
    self.decoder = []

    self.down_sampling_features = []
    
    # encoder
    self.encoder0 = self.conv_layer(in_channel=1, out_channel=32, maxpool=True, kernel_size=3, padding=1, name='conv_%d' % 0,)
    self.maxpool0 = self.batchnorm_maxpool_relu(32)

    self.encoder1 = self.conv_layer(in_channel=32, out_channel=64, maxpool=True, kernel_size=3, padding=1, name='conv_%d' % 1,)
    self.maxpool1 = self.batchnorm_maxpool_relu(64)

    self.encoder2 = self.conv_layer(in_channel=64, out_channel=128, maxpool=True, kernel_size=3, padding=1, name='conv_%d' % 2,)
    self.maxpool2 = self.batchnorm_maxpool_relu(128)

    self.encoder3 = self.conv_layer(in_channel=128, out_channel=256, maxpool=True, kernel_size=3, padding=1, name='conv_%d' % 3,)
    self.maxpool3 = self.batchnorm_maxpool_relu(256)

    self.encoder4 = self.conv_layer(in_channel=256, out_channel=256, maxpool=True, kernel_size=3, padding=1, name='conv_%d' % 4,)
    self.maxpool4 = self.batchnorm_maxpool_relu(256)

    # bottleneck
    self.bottleneck = self.conv_layer(in_channel=256, out_channel=512, maxpool=False, kernel_size=3, padding=1,)

    # decoder
    self.upconv0 = self.upconv_layer(in_channel=512, out_channel=512, kernel_size=2, stride= 2, name="")
    self.upconv1 = self.upconv_layer(in_channel=256, out_channel=256, kernel_size=2, stride=2, name="")
    self.upconv2 = self.upconv_layer(in_channel=128, out_channel=128, kernel_size=2, stride=2, name="")
    self.upconv3 = self.upconv_layer(in_channel=64, out_channel=64, kernel_size=2, stride=2, name="")
    self.upconv4 = self.upconv_layer(in_channel=32, out_channel=32, kernel_size=2, stride=2, name="")

    self.decoder0 = Res_Block(in_channels=768, out_channels=256)
    self.decoder1 = Res_Block(in_channels=512, out_channels=128)
    self.decoder2 = Res_Block(in_channels=256, out_channels=64)
    self.decoder3 = Res_Block(in_channels=128, out_channels=32)
    self.decoder4 = self.conv_layer_out(in_channel=64, out_channel=1, kernel_size=1, padding=0)

    self.branch_output0 = nn.Conv3d(in_channels=128, out_channels=1, padding=0, kernel_size=1)
    self.branch_output1 = nn.Conv3d(in_channels=64, out_channels=1, padding=0, kernel_size=1)
    self.branch_output2 = nn.Conv3d(in_channels=32, out_channels=1, padding=0, kernel_size=1)

    if pre_trained:
        pre_trained_weight = torch.load('/weights/run_20190719_00_epoch_best_mae.p', map_location=torch.device('cpu'))
        model_dict = self.state_dict()
        weights = list(pre_trained_weight.values())
        for i, w in enumerate(model_dict.keys()):
            if i == 35:
                break
            assert model_dict[w].shape == weights[i].shape
            model_dict[w] = weights[i]
    
        self.load_state_dict(model_dict)

  @staticmethod
  def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2, name=""):
    layer = nn.Sequential(
              nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
          )
    return layer

  @staticmethod
  def conv_layer_out(in_channel, out_channel, kernel_size=3, padding=0):
    layer = nn.Sequential(
              nn.Conv3d(in_channel, in_channel, padding=1, kernel_size=3),
              nn.BatchNorm3d(in_channel),
              nn.LeakyReLU(),
              
              nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
              nn.Sigmoid(),
          )
    return layer
  
  @staticmethod
  def batchnorm_maxpool_relu(out_channel):
    layer = nn.Sequential(
      nn.BatchNorm3d(out_channel),
      nn.MaxPool3d(2, stride=2),
      nn.LeakyReLU(),
    )
    return layer
    
  @staticmethod
  def upconv_layer(in_channel, out_channel, kernel_size=2, stride=2, padding=0, name=""):
    layer = nn.Sequential(nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2 ))
    return layer
  
  def forward(self, x):
    x_0_feature = self.encoder0(x)
    x_0_maxpool = self.maxpool0(x_0_feature)
    
    x_1_feature = self.encoder1(x_0_maxpool)
    x_1_maxpool = self.maxpool1(x_1_feature)

    x_2_feature = self.encoder2(x_1_maxpool)
    x_2_maxpool = self.maxpool2(x_2_feature)

    x_3_feature = self.encoder3(x_2_maxpool)
    x_3_maxpool = self.maxpool3(x_3_feature)

    x_4_feature = self.encoder4(x_3_maxpool)
    x_4_maxpool = self.maxpool4(x_4_feature)
    
    x_5_feature = self.bottleneck(x_4_maxpool)
    x_5_up = self.upconv0(x_5_feature)

    x_6_feature = self.decoder0(torch.cat((x_4_feature, x_5_up), dim=1))
    x_6_up = self.upconv1(x_6_feature)

    x_7_feature = self.decoder1(torch.cat((x_3_feature, x_6_up), dim=1))
    x_7_up = self.upconv2(x_7_feature)

    x_8_feature = self.decoder2(torch.cat((x_2_feature, x_7_up), dim=1))
    x_8_up = self.upconv3(x_8_feature)
    
    x_9_feature = self.decoder3(torch.cat((x_1_feature, x_8_up), dim=1))
    x_9_up = self.upconv4(x_9_feature)

    x_out = self.decoder4(torch.cat((x_0_feature, x_9_up), dim=1))

    branch_output0 = self.branch_output0(x_7_feature)
    branch_output1 = self.branch_output1(x_8_feature)
    branch_output2 = self.branch_output2(x_9_feature)
    return branch_output0, branch_output1, branch_output2, x_out

if __name__ == '__main__':
    test_input = torch.rand(1, 1, 160, 192, 160)
    temp_model = Model()

    test_out = temp_model(test_input)
    for each in test_out:
        print(each.shape)

