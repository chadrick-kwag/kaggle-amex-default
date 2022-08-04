import torch


class Encoder(torch.nn.Module):
    def __init__(
        self, d_model, nhead, activation, num_layers, dropout=0, layer_norm_eps=1e-5
    ) -> None:

        assert num_layers > 0, "invalid num layers"

        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, activation=activation, dropout=dropout
        )

        encoder_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, norm=encoder_norm
        )

    def forward(self, x, src_key_padding_mask=None):

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
