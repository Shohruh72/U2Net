import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n=1):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=n, dilation=n)
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RSU(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, depth):
        super(RSU, self).__init__()
        self.hx_dec = None
        self.encoder = torch.nn.ModuleList([Conv(in_ch, out_ch, n=1)] +
                                           [Conv(out_ch, mid_ch, n=1)] +
                                           [Conv(mid_ch, mid_ch, n=1) for _ in range(depth - 3)])

        self.mid_conv = torch.nn.ModuleList([Conv(mid_ch, mid_ch, n=1)] +
                                            [Conv(mid_ch, mid_ch, n=2)])

        self.pools = torch.nn.ModuleList([torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) for _ in range(depth - 2)])

        self.decoder = torch.nn.ModuleList([Conv(mid_ch * 2, mid_ch, n=1) for _ in range(depth - 2)] +
                                           [Conv(mid_ch * 2, out_ch, n=1)])

    def forward(self, x):
        global hx_dec
        hx = x
        hxin = self.encoder[0](hx)

        features = {'hx1': self.encoder[1](hxin)}
        for i in range(2, len(self.encoder)):
            hx = self.pools[i - 2](features[f'hx{i - 1}'])
            features[f'hx{i}'] = self.encoder[i](hx)

        hx_mid = self.pools[-1](features[f'hx{len(self.encoder) - 1}'])
        features['hx_mid'] = self.mid_conv[0](hx_mid)
        features['hx_mid2'] = self.mid_conv[1](features['hx_mid'])

        hx_up = features['hx_mid2']
        for i in range(len(self.decoder)):
            if i == 0:
                hx_cat = torch.cat((features['hx_mid2'], features['hx_mid']), dim=1)
            else:
                hx_cat = torch.cat((hx_up, features[f'hx{len(self.encoder) - i}']), dim=1)
            hx_dec = self.decoder[i](hx_cat)
            if i < len(self.decoder) - 1:
                hx_up = F.interpolate(hx_dec, size=features[f'hx{len(self.encoder) - i - 1}'].shape[2:],
                                      mode='bilinear')

        return hx_dec + hxin


class RSU4F(torch.nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.conv0 = Conv(in_ch, out_ch, n=1)

        self.conv1 = Conv(out_ch, mid_ch, n=1)
        self.conv2 = Conv(mid_ch, mid_ch, n=2)
        self.conv3 = Conv(mid_ch, mid_ch, n=4)

        self.conv4 = Conv(mid_ch, mid_ch, n=8)

        self.conv3d = Conv(mid_ch * 2, mid_ch, n=4)
        self.conv2d = Conv(mid_ch * 2, mid_ch, n=2)
        self.conv1d = Conv(mid_ch * 2, out_ch, n=1)

    def forward(self, x):
        hx = x

        hxin = self.conv0(hx)

        hx1 = self.conv1(hxin)
        hx2 = self.conv2(hx1)
        hx3 = self.conv3(hx2)

        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.conv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.conv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2NETP(torch.nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU(in_ch, 16, 64, 7)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(64, 16, 64, 6)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(64, 16, 64, 5)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(64, 16, 64, 4)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU(128, 16, 64, 4)
        self.stage3d = RSU(128, 16, 64, 5)
        self.stage2d = RSU(128, 16, 64, 6)
        self.stage1d = RSU(128, 16, 64, 7)

        self.side1 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = torch.nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = F.upsample(hx6, size=hx5.shape[2:], mode='bilinear')

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.upsample(hx5d, size=hx4.shape[2:], mode='bilinear')

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.upsample(hx4d, size=hx3.shape[2:], mode='bilinear')

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.upsample(hx3d, size=hx2.shape[2:], mode='bilinear')

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.upsample(hx2d, size=hx1.shape[2:], mode='bilinear')

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = F.upsample(d2, size=d1.shape[2:], mode='bilinear')

        d3 = self.side3(hx3d)
        d3 = F.upsample(d3, size=d1.shape[2:], mode='bilinear')

        d4 = self.side4(hx4d)
        d4 = F.upsample(d4, size=d1.shape[2:], mode='bilinear')

        d5 = self.side5(hx5d)
        d5 = F.upsample(d5, size=d1.shape[2:], mode='bilinear')

        d6 = self.side6(hx6)
        d6 = F.upsample(d6, size=d1.shape[2:], mode='bilinear')

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
