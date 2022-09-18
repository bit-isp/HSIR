"""
Ported from official caffe implementation
https://github.com/qzhang95/HSID-CNN
"""

import torch
import torch.nn as nn


def hsid_cnn():
    return HSIDCNN()


def hsid_cnn_fast():
    return HSIDCNN(fast_impl=True)

def hsid_cnn_fast_b6():
    return HSIDCNN(fast_impl=True, fast_bs=6)

def hsid_cnn_fast_b12():
    return HSIDCNN(fast_impl=True, fast_bs=12)

class HSIDCNN(nn.Module):
    def __init__(self, num_adj_bands=12, fast_impl=False, fast_bs=4):
        super().__init__()
        self.num_adj_bands = num_adj_bands
        self.fast_impl = fast_impl
        self.fast_bs = fast_bs
        
        self.conv_k3 = nn.Conv2d(num_adj_bands * 2, 20, 3, 1, 1)
        self.conv_k5 = nn.Conv2d(num_adj_bands * 2, 20, 5, 1, 2)
        self.conv_k7 = nn.Conv2d(num_adj_bands * 2, 20, 7, 1, 3)

        self.conv_k3_2 = nn.Conv2d(1, 20, 3, 1, 1)
        self.conv_k5_2 = nn.Conv2d(1, 20, 5, 1, 2)
        self.conv_k7_2 = nn.Conv2d(1, 20, 7, 1, 3)

        self.conv1 = nn.Conv2d(120, 60, 3, 1, 1)
        self.conv2 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv3 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv4 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv5 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv6 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv7 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv8 = nn.Conv2d(60, 60, 3, 1, 1)
        self.conv9 = nn.Conv2d(60, 60, 3, 1, 1)

        self.tail = nn.Sequential(
            nn.Conv2d(60 * 4, 15, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(15, 1, 3, 1, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        num_bands = x.shape[1]
        outputs = []

        inputs_x, inputs_adj_x = [], []
        for i in range(self.num_adj_bands):
            inputs_x.append(x[:, i:i + 1, :, :])
            inputs_adj_x.append(x[:, :self.num_adj_bands * 2, :, :])

        for i in range(self.num_adj_bands, num_bands - self.num_adj_bands):
            adj = torch.cat([x[:, i - self.num_adj_bands:i, :, :],
                             x[:, i + 1:i + 1 + self.num_adj_bands, :, :]], dim=1)
            inputs_x.append(x[:, i:i + 1, :, :])
            inputs_adj_x.append(adj)

        for i in range(num_bands - self.num_adj_bands, num_bands): 
            inputs_x.append(x[:, i:i + 1, :, :])
            inputs_adj_x.append(x[:, -self.num_adj_bands * 2:, :, :])

        if self.fast_impl:
            batch_size = x.shape[0]
            inputs_x = torch.cat(inputs_x, dim=0)
            inputs_adj_x = torch.cat(inputs_adj_x, dim=0)
            outputs = []
            for i in range(0,num_bands,self.fast_bs):
                output = self._forward(inputs_x[i:i+self.fast_bs], inputs_adj_x[i:i+self.fast_bs])
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            outputs_batch = []
            for i in range(batch_size):
                outputs_batch.append(outputs[i::batch_size])
            return torch.cat(outputs_batch, dim=0).transpose(0,1)
        else:
            for i in range(num_bands):
                output = self._forward(inputs_x[i], inputs_adj_x[i])
                outputs.append(output)
            return torch.cat(outputs, dim=1)
        
    
    def _forward(self, x, adj_x):
        feat3 = self.conv_k3(adj_x)
        feat5 = self.conv_k5(adj_x)
        feat7 = self.conv_k7(adj_x)
        feat_3_5_7 = torch.cat([feat3, feat5, feat7], dim=1).relu()

        feat3_2 = self.conv_k3_2(x)
        feat5_2 = self.conv_k5_2(x)
        feat7_2 = self.conv_k7_2(x)
        feat_3_5_7_2 = torch.cat([feat3_2, feat5_2, feat7_2], dim=1).relu()

        feat_all = torch.cat([feat_3_5_7, feat_3_5_7_2], dim=1)

        tmp = self.conv1(feat_all).relu()
        tmp = self.conv2(tmp).relu()
        feat_conv3 = self.conv3(tmp).relu()

        tmp = self.conv4(feat_conv3).relu()
        feat_conv5 = self.conv5(tmp).relu()

        tmp = self.conv6(feat_conv5).relu()
        feat_conv7 = self.conv7(tmp).relu()

        tmp = self.conv8(feat_conv7).relu()
        feat_conv9 = self.conv9(tmp).relu()

        feat_all = torch.cat([feat_conv3, feat_conv5, feat_conv7, feat_conv9], dim=1)
        out = self.tail(feat_all)

        return out


if __name__ == '__main__':
    net = HSIDCNN().cuda()
    x = torch.randn(4,31,64,64).cuda()
    out = net(x)
    print(out.shape)
    