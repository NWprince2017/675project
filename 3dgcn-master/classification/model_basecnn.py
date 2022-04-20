import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))

        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1),
            nn.ReLU())

        self.fc1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 40)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=20)

        x = self.layer1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=20)
        x = self.layer2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=20)
        x = self.layer3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=20)
        x = self.layer4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x4, k=10)
        x = self.layer5(x4)
        x5 = x.max(dim=-1, keepdim=False)[0]

        out = x5.reshape(x5.size(0), -1)
        out = self.dropout(out)

        out = self.fc1(out)
        pred = self.fc2(out)
        return pred


def test():
    import time
    sys.path.append("..")
    from util import parameter_number

    device = torch.device('cuda:0')
    points = torch.zeros(8, 1024, 3).to(device)
    model = BaseCNN().to(device)
    start = time.time()
    output = model(points)

    print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    print("Inputs size: {}".format(points.size()))
    print("Output size: {}".format(output.size()))


if __name__ == '__main__':
    test()
