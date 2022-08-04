import torch


class TwoStageLinearModule(torch.nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(input_dim, intermediate_dim)
        self.l2 = torch.nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):

        y = self.l1(x)
        # y = torch.relu(y)
        y = torch.nn.functional.gelu(y)
        y = self.l2(y)
        return torch.nn.functional.gelu(y)
        # return torch.relu(y)
