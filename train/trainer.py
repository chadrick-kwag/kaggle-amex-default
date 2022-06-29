import pytorch_lightning as pl
import torch

from dataset.no_nan_dataset import NoNanColsDataset_v1


class Encoder(torch.nn.Module):
    def __init__(
        self, d_model, nhead, activation, num_layers, layer_norm_eps=1e-5
    ) -> None:

        assert num_layers > 0, "invalid num layers"

        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, activation=activation
        )

        encoder_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, norm=encoder_norm
        )


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


class Model_v1(pl.LightningModule):
    def __init__(
        self,
        d_model,
        nhead,
        activation,
        num_layers,
        token_type_embedding_size,
        feature_size,
    ):
        super().__init__()

        self.encoder = Encoder(d_model, nhead, activation, num_layers)

        self.token_type_embedding = torch.nn.Embedding(
            token_type_embedding_size, d_model
        )
        self.token_feature_embedding = TwoStageLinearModule(feature_size, 512, d_model)

        self.encoder_output_clsf_module = TwoStageLinearModule(d_model, 512, 2)

    def forward(self, token_types, token_features):

        token_type_embed = self.token_type_embedding(token_types)
        token_feature_embed = self.token_feature_embedding(token_features)

        transform_input = token_type_embed + token_feature_embed

        encoder_output = self.encoder(transform_input)

        clsf_output = self.encoder_output_clsf_module(encoder_output)

        return {"raw_encoder_output": encoder_output, "clsf_output": clsf_output}

    def training_step(self, batch, batch_idx):

        return


def main():

    trainer = pl.Trainer()
