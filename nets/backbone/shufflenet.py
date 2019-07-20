import torch as t
import torch.nn as nn
import math
from collections import OrderedDict

__all__ = ['shufflenet2']

#### The model below is defined by myself


def channel_shuffle(x, groups=2):
  bat_size, channels, w, h = x.shape
  group_c = channels // groups
  x = x.view(bat_size, groups, group_c, w, h)
  x = t.transpose(x, 1, 2).contiguous()
  x = x.view(bat_size, -1, w, h)
  return x

# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )

def conv_bn(in_c, out_c, stride=2):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )


class ShuffleBlock(nn.Module):
  def __init__(self, in_c, out_c, downsample=False):
    super(ShuffleBlock, self).__init__()
    self.downsample = downsample
    half_c = out_c // 2
    if downsample:
      self.branch1 = nn.Sequential(
          # 3*3 dw conv, stride = 2
          nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
          nn.BatchNorm2d(in_c),
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
      
      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 2
          nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
    else:
      # in_c = out_c
      assert in_c == out_c
        
      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 1
          nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
      
      
  def forward(self, x):
    out = None
    if self.downsample:
      # if it is downsampling, we don't need to do channel split
      out = t.cat((self.branch1(x), self.branch2(x)), 1)
    else:
      # channel split
      channels = x.shape[1]
      c = channels // 2
      x1 = x[:, :c, :, :]
      x2 = x[:, c:, :, :]
      out = t.cat((x1, self.branch2(x2)), 1)
    return channel_shuffle(out, 2)
    

class ShuffleNet2(nn.Module):
  def __init__(self, input_size=416, net_type=1):
    super(ShuffleNet2, self).__init__()
    assert input_size % 32 == 0 # 因为一共会下采样32倍
    self.layers_out_filters = [24, 116, 232, 1024] # used for shufflenet v2
    
    self.stage_repeat_num = [4, 8, 4]
    if net_type == 0.5:
      self.out_channels = [3, 24, 48, 96, 192, 1024]
    elif net_type == 1:
      self.out_channels = [3, 24, 116, 232, 464, 1024]
    elif net_type == 1.5:
      self.out_channels = [3, 24, 176, 352, 704, 1024]
    elif net_type == 2:
      self.out_channels = [3, 24, 244, 488, 976, 2948]
    elif net_type == -1:
      self.out_channels = [3, 24, 128, 256, 512, 1024]
    else:
      print("the type is error, you should choose 0.5, 1, 1.5 or 2")
      
    # let's start building layers
    self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    in_c = self.out_channels[1]
    
    self.stage2 = []
    self.stage3 = []
    self.stage4 = []
    for stage_idx in range(len(self.stage_repeat_num)):
      out_c = self.out_channels[2+stage_idx]
      repeat_num = self.stage_repeat_num[stage_idx]
      stage = []
      for i in range(repeat_num):
        if i == 0:
          stage.append(ShuffleBlock(in_c, out_c, downsample=True))
        else:
          stage.append(ShuffleBlock(in_c, in_c, downsample=False))
        in_c = out_c
      if stage_idx == 0:
        self.stage2 = stage
      elif stage_idx == 1:
        self.stage3 = stage
      elif stage_idx == 2:
        self.stage4 = stage
      else:
        print("error")
    # self.stages = nn.Sequential(*self.stages)
    self.stage2 = nn.Sequential(*self.stage2) # 58 * 58 * 116
    self.stage3 = nn.Sequential(*self.stage3) # 26 * 26 * 232
    self.stage4 = nn.Sequential(*self.stage4)
    in_c = self.out_channels[-2]
    out_c = self.out_channels[-1]
    self.conv5 = conv_1x1_bn(in_c, out_c, 1) # 13 * 13 * 1024
    # self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(input_size/32)) # 如果输入的是224，则此处为7
    
    # # fc layer
    # self.fc = nn.Linear(out_c, num_classes)
    

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    out3 = self.stage2(x)
    out4 = self.stage3(out3)
    out5 = self.stage4(out4)
    out5 = self.conv5(out5)
    # x = self.g_avg_pool(x)
    # x = x.view(-1, self.out_channels[-1])
    # x = self.fc(x)
    return out3, out4, out5

def shufflenet2(pretrained, **kwargs):
    """Constructs a darknet-53 model.
    """
    model = ShuffleNet2()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(t.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
