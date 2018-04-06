from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch

class RegionGaussian(nn.Module):
    def __init__(self, region_size = 3, adopt_concat = True):
        super(RegionGaussian, self).__init__()
        self.region_size = (region_size - 1) // 2
        self.adopt_concat = adopt_concat

    def forward(self, x):
        if self.region_size == 0:
            if self.adopt_concat:
                return torch.cat([x, torch.exp(x)], dim = 1)
            else:
                return torch.exp(x)
        batch, channel, height, width = x.size()
        x_square = x ** 2
        x_exp = torch.exp(x)
        x_square_exp = torch.exp(x_square)
        x_diff = x_square_exp / (2 * x_exp)
        result = torch.zeros_like(x)
        for h in range(height):
            for w in range(width):
                height_up = max(0, min(height-1, h - self.region_size))
                height_down = max(0, min(height-1, h + self.region_size))
                width_up = max(0, min(width-1, w - self.region_size))
                width_down = max(0, min(width-1, w + self.region_size))
                _num = (height_down - height_up + 1) * (width_down - width_up + 1)
                _tensor = x_diff[:, :, height_up:height_down+1, width_up:width_down+1] * x_exp[:, :, h:h+1, w:w+1]
                _tensor = _tensor.mean(-1).mean(-1)
                result[:, :, h, w] = _tensor
        if self.adopt_concat:
            return torch.cat([x, result], dim = 1)
        else:
            return result

if __name__ == '__main__':
    net = RegionGaussian()
    input1 = np.asarray(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]
    )
    input2 = np.asarray(
        [[[[0.095, 0.205, 0.305], [0.395, 0.505, 0.605], [0.705, 0.795, 0.895]]]]
    )
    # print(input2 - input1)
    input1 = Variable(torch.from_numpy(input1).float())
    result1 = net(input1)
    input2 = Variable(torch.from_numpy(input2).float())
    result2 = net(input2)
    print(result1 - result2)
