import torch
from model.encoder import Encoder
from model.final_layer import TwoStageLinearModule


class Model_v1(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        activation,
        num_layers,
        token_type_embedding_size,
        feature_dim,
    ):
        self.encoder = Encoder(d_model, nhead, activation, num_layers)

        self.token_type_embedding = torch.nn.Embedding(
            token_type_embedding_size, d_model
        )
        self.token_feature_embedding = TwoStageLinearModule(feature_dim, 512, d_model)

        self.encoder_output_clsf_module = TwoStageLinearModule(d_model, 512, 2)


class Model_v2(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        activation,
        num_layers,
        dim_feedforward,
        embedding_config_list: list,
        dropout=None,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model, n_head, activation, num_layers, dim_feedforward, dropout=dropout
        )

        self.encoder_output_clsf_module = TwoStageLinearModule(d_model, 512, 2)
        self.embedding_config_list = embedding_config_list

        self.embedding_list = []
        self.colname_to_index_dict = {}

        for i, c in enumerate(embedding_config_list):
            if c["type"] == "float":
                embedding = torch.nn.Linear(2, d_model)
            elif c["type"] == "category":
                embedding = torch.nn.Linear(c["size"] + 1, d_model)
            else:
                raise Exception(f"invalid embedding config: {c}")

            setattr(self, f"embedding_{i}", embedding)
            self.embedding_list.append(embedding)
            self.colname_to_index_dict[c["col_name"]] = i

    def forward(self, data, key_padding_mask):
        """
        x format: {
            'data': ...,
            'key_padding_mask': tensor.bool
        }

        """

        # get embedded feature and added them up
        aggregated_embedded_feature = None

        for k, v in data.items():
            cat_vector = torch.cat((v["na"], v["value"]), dim=-1)
            index = self.colname_to_index_dict[k]
            embedding_net = self.embedding_list[index]
            embedding_vector = embedding_net(cat_vector)

            if aggregated_embedded_feature is None:
                aggregated_embedded_feature = embedding_vector
            else:
                aggregated_embedded_feature += embedding_vector
        encoder_input = aggregated_embedded_feature.permute(1, 0, 2)
        encoder_output = self.encoder(
            encoder_input, src_key_padding_mask=key_padding_mask
        )

        clsf_output = self.encoder_output_clsf_module(encoder_output)

        encoder_output = encoder_output.transpose(0, 1)
        clsf_output = clsf_output.transpose(0, 1)

        return {"encoder_output": encoder_output, "clsf_output": clsf_output}
