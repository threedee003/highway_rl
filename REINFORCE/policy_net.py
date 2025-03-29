import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim* input_dim, input_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim//2, input_dim // 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim//5, input_dim//5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim//5, output_dim)
        )


    def forward(self, x) -> torch.Tensor:
        x = x.flatten().view(1,-1)
        x = self.sequential(x)
        return F.softmax(x, dim=1)




if __name__ == '__main__':
    x = torch.randn(5,5)
    print(x.shape)
    model = PolicyNet(input_dim=5, output_dim=5)
    y = model(x)
    print(y.shape)