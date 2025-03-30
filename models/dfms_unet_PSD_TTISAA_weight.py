import torch
import torch.nn as nn
from .utils import InitWeights_He
import torch.nn.functional as F

class SpaceToDepth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # 确保输入张量的 height 和 width 都是偶数
         if x.size(-2) % 2 != 0 or x.size(-1) % 2 != 0:
            raise ValueError("The height and width of the input tensor must be even.")
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    # Coordinate Attention for Efficient Mobile Network Design
    '''
        Recent studies on mobile network design have demonstrated the remarkable effectiveness of channel attention (e.g., the Squeeze-and-Excitation attention) for lifting
    model performance, but they generally neglect the positional information, which is important for generating spatially selective attention maps. In this paper, we propose a
    novel attention mechanism for mobile iscyy networks by embedding positional information into channel attention, which
    we call “coordinate attention”. Unlike channel attention
    that transforms a feature tensor to a single feature vector iscyy via 2D global pooling, the coordinate attention factorizes channel attention into two 1D feature encoding
    processes that aggregate features along the two spatial directions, respectively
    '''

    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()   #（1,32,48,48）
        pool_h = nn.AdaptiveAvgPool2d((h, 1))   #(48,1)
        pool_w = nn.AdaptiveAvgPool2d((1, w))  #AdaptiveAvgPool2d(output_size=(1, 48))
        x_h = pool_h(x)    #(1,32,48,1)
        x_w = pool_w(x).permute(0, 1, 3, 2)  #(1,32,48,1)

        y = torch.cat([x_h, x_w], dim=2)   #(1,32,96,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()

        self.sab = CA(in_channels, in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = (sab + cab)
        return out


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1 + x2 + x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1,
                      padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))
        self.space_to_depth = SpaceToDepth()
        self.conv11 = nn.Conv2d(
            out_c *4, out_c, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        x_down = self.down(x)     #x=(1,32,48,48)
        x_down = self.space_to_depth(x_down)
        x_down = self.conv11(x_down)
        return x_down   #(1,64,24,24)


class block(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_up=False, is_down=False, fuse=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c // 2)
        if self.is_down == True:
            self.down = down(out_c, out_c * 2)


    def forward(self, x):
        if self.in_c != self.out_c:
            x = self.fuse(x)     #x=(1,1,48,48)
        x = self.conv(x)   #x=(1,32,48,48)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class FR_UNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=1, feature_scale=2, dropout=0.2, fuse=True, out_ave=True):
        super(FR_UNet, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.weights = nn.Parameter(torch.ones(5))  # 可学习的权重参数
        self.block1_3 = block(
            num_channels, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse)

        self.block11 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1] * 2, filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)

        self.block20 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        #self.ca3_1 = CA(filters[3], filters[3])

        self.block30 = block(
            filters[2] * 2, filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2] * 3, filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse)
        #self.block40 = block(filters[3], filters[3],
        #                     dp=dropout, is_up=True, is_down=False, fuse=fuse)

        self.affinity_attention = AffinityAttention(128)
        #self.ca40 = CA(filters[2], filters[2])
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        #x_down3_1 = self.ca3_1(x_down3_1)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        #_, x_up40 = self.block40(x_down3_1)

        # Do Attenttion operations here
        x_up40 = self.affinity_attention(x3_1)

        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))

        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))

        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))


        if self.out_ave == True:

            # 应用softmax函数来归一化权重，确保权重和为1
            weights = F.softmax(self.weights, dim=0)
            # 计算每个分支的输出
            output1 = self.final1(x1_1)
            output2 = self.final2(x10)
            output3 = self.final3(x11)
            output4 = self.final4(x12)
            output5 = self.final5(x13)
           
            # 使用归一化后的权重计算加权平均输出
            output = (output1 * weights[0] + output2 * weights[1] +
                      output3 * weights[2] + output4 * weights[3] + output5 * weights[4])
            """
            output = (self.final1(x1_1) + self.final2(x10) +
                      self.final3(x11) + self.final4(x12) + self.final5(x13)) / 5
            """
        else:
            output = self.final5(x13)

        return output
