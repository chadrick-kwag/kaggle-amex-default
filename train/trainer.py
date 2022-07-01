import pytorch_lightning as pl, os, yaml, sys, datetime
import torch, argparse
from munch import munchify

if __name__ == "__main__":

    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    if rootdir not in sys.path:
        sys.path.append(rootdir)

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

    def forward(self, x, src_key_padding_mask=None):

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


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
        config,
        d_model,
        nhead,
        activation,
        num_layers,
        token_type_embedding_size,
        feature_dim,
    ):
        super().__init__()
        self.config = config
        self.batch_size = 1

        self.encoder = Encoder(d_model, nhead, activation, num_layers)

        self.token_type_embedding = torch.nn.Embedding(
            token_type_embedding_size, d_model
        )
        self.token_feature_embedding = TwoStageLinearModule(feature_dim, 512, d_model)

        self.encoder_output_clsf_module = TwoStageLinearModule(d_model, 512, 2)

    def forward(self, token_types, token_features, src_key_padding_mask):

        token_type_embed = self.token_type_embedding(token_types)
        token_feature_embed = self.token_feature_embedding(token_features)

        transform_input = token_type_embed + token_feature_embed

        transform_input = transform_input.permute(1, 0, 2)

        encoder_output = self.encoder(
            transform_input, src_key_padding_mask=src_key_padding_mask
        )

        clsf_output = self.encoder_output_clsf_module(encoder_output)

        # encoder output shp: (S,B,D), clsf_output shp: (S,B,D')
        encoder_output = encoder_output.transpose(0, 1)
        clsf_output = clsf_output.transpose(0, 1)
        return {"raw_encoder_output": encoder_output, "clsf_output": clsf_output}

    def validation_step(self, batch, batch_idx):

        input_data = {}
        input_data["token_types"] = batch["token_types"]
        input_data["token_features"] = batch["token_features"]
        input_data["src_key_padding_mask"] = batch["key_padding_mask"]

        output = self.forward(**input_data)

        cls_output = output["clsf_output"][:, 0]
        cls_output = torch.softmax(cls_output, -1)

        default_prob = cls_output[:, None, 1]

        loss = torch.nn.functional.binary_cross_entropy(default_prob, batch["label"])

        self.log("valid_loss", loss)

    def training_step(self, batch, batch_idx):

        input_data = {}
        input_data["token_types"] = batch["token_types"]
        input_data["token_features"] = batch["token_features"]
        input_data["src_key_padding_mask"] = batch["key_padding_mask"]

        output = self.forward(**input_data)

        cls_output = output["clsf_output"][:, 0]
        cls_output = torch.softmax(cls_output, -1)

        default_prob = cls_output[:, None, 1]

        loss = torch.nn.functional.binary_cross_entropy(default_prob, batch["label"])

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer


def main(config):
    model = Model_v1(
        config,
        config.model.d_model,
        config.model.n_head,
        config.model.activation,
        config.model.num_layers,
        config.model.token_type_embedding_size,
        config.model.feature_dim,
    )
    train_dataset = NoNanColsDataset_v1(
        config.train_data.dir_list,
        config.train_data.label_csv_file,
        config.train_data.data_flatten_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=NoNanColsDataset_v1.collate,
    )

    valid_dataset = NoNanColsDataset_v1(
        config.valid_data.dir_list,
        config.valid_data.label_csv_file,
        config.valid_data.data_flatten_size,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        collate_fn=NoNanColsDataset_v1.collate,
    )

    # create outputdir
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f"ckpt/trainer/{timestamp}"
    os.makedirs(outputdir)

    # save config
    p = os.path.join(outputdir, "usedconfig.yaml")

    with open(p, "w") as fd:
        yaml.dump(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        # auto_scale_batch_size="power",
        val_check_interval=config.val_check_interval,
        default_root_dir=outputdir,
    )
    # trainer.tune(model)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="config file")

    args = parser.parse_args()

    config = args.config
    assert os.path.exists(config), "config file not exist"

    with open(config, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    config = munchify(config)

    main(config)


if __name__ == "__main__":

    cli()
