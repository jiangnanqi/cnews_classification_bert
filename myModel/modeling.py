import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self):
        super(Self_Attn, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        proj_query = x.permute(0, 2, 1)  # B X CX(N)
        proj_key = x  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        out = torch.bmm(x, attention.permute(0, 2, 1))

        return out

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, in_plane, out_plane, stride = 1):
        super(BasicBlock, self).__init__()
        if in_plane != out_plane:
            stride = 2
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_plane, in_plane, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_plane, out_plane, kernel_size=3, stride=stride, padding=1),
        )
        self.in_plane = in_plane
        self.out_plane = out_plane
        self.downSample = nn.Conv1d(in_plane, out_plane, kernel_size=3, stride=stride, padding=1) \
            if self.stride == 2 else None
    def forward(self, input):
        x = self.conv1(input)
        if self.stride == 1:
            y =  x + input
        else:
            y = self.downSample(input) + x
        return F.relu(y)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        num = 2
        self.list = nn.ModuleList(
            nn.Sequential(
                BasicBlock(768, 768),
                BasicBlock(768 , 768, stride=2)
            ) for i in range(4)
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
    def forward(self, input):
        x = input
        for it in self.list:
            x = it(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    a = torch.rand([4, 768, 64])
    net = ResNet()
    a, net = a.cuda(), net.cuda()
    b = net(a)
    print(b.size())