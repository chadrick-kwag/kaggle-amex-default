import torch


class TwoStageLinearModule(torch.nn.Module):
    def __init__(
        self, input_dim, intermediate_dim, output_dim, last_activation=True
    ) -> None:
        super().__init__()

        self.last_activation = last_activation

        self.l1 = torch.nn.Linear(input_dim, intermediate_dim)
        self.l2 = torch.nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):

        y = self.l1(x)
        y = torch.nn.functional.gelu(y)
        y = self.l2(y)

        if self.last_activation:
            return torch.nn.functional.gelu(y)
        else:
            return y
