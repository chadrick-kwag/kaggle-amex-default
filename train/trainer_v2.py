""" 
compared to `trainer.py` separate model building
"""


import pytorch_lightning as pl, os, yaml, sys, datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import torch, argparse, torchmetrics
from munch import munchify


if __name__ == "__main__":

    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    if rootdir not in sys.path:
        sys.path.append(rootdir)

from dataset.no_nan_dataset import NoNanColsDataset_v1
from dataset.na_flag_dataset import NA_Flag_Dataset
from model.model import Model_v2


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.batch_size = 1

        self.auroc_metric = torchmetrics.AUROC()

        self.build_model()

    def build_model(self):

        model_config = self.config.model_config

        assert os.path.exists(model_config.embedding_config_list)

        with open(model_config.embedding_config_list, "r") as fd:
            data = yaml.load(fd, Loader=yaml.FullLoader)
            model_embedding_config_list = data["model_embedding_config_list"]

        model_kwargs = vars(model_config)
        model_kwargs["embedding_config_list"] = model_embedding_config_list

        self.model = Model_v2(**model_kwargs)

    def forward(self, data, key_padding_mask):

        return self.model(data, key_padding_mask)

    def validation_step(self, batch, batch_idx):

        input_data = {}
        input_data["data"] = batch["data"]
        input_data["key_padding_mask"] = batch["key_padding_mask"]

        output = self.forward(**input_data)

        cls_output = output["clsf_output"][:, 0]
        cls_output = torch.softmax(cls_output, -1)

        default_prob = cls_output[:, None, 1]

        loss = torch.nn.functional.binary_cross_entropy(default_prob, batch["label"])

        self.auroc_metric.update(
            default_prob.cpu().view(-1), batch["label"].cpu().int().view(-1)
        )

        self.log("valid_loss", loss)

    def validation_epoch_end(self, outputs) -> None:

        auroc = self.auroc_metric.compute()
        print(f"validation auroc: {auroc}")
        self.log("auroc", auroc)

    def training_step(self, batch, batch_idx):

        input_data = {}
        input_data["data"] = batch["data"]
        input_data["key_padding_mask"] = batch["key_padding_mask"]

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
    model = ModelWrapper(config)

    # train_dataset = NoNanColsDataset_v1(
    #     config.train_data.dir_list,
    #     config.train_data.label_csv_file,
    #     config.train_data.data_flatten_size,
    # )
    train_dataset = NA_Flag_Dataset(
        config.train_data.dir_list,
        config.train_data.parsing_config_list_file,
        config.train_data.label_sqlite3_file,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
        num_workers=config.train_data.num_workers,
        prefetch_factor=config.train_data.prefetch_factor,
        drop_last=config.train_data.drop_last,
    )

    valid_dataset = NA_Flag_Dataset(
        config.valid_data.dir_list,
        config.valid_data.parsing_config_list_file,
        config.valid_data.label_sqlite3_file,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        collate_fn=valid_dataset.collate,
        num_workers=2,
        prefetch_factor=2,
    )

    # create outputdir
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    suffix = config.suffix
    outputdir = f"ckpt/trainer_v2/{timestamp}"
    if suffix is not None:
        outputdir = outputdir + f"-{suffix}"
    os.makedirs(outputdir)

    # save config
    p = os.path.join(outputdir, "usedconfig.yaml")

    with open(p, "w") as fd:
        yaml.dump(config, fd)

    # callback setup

    callbacks = []
    savedir = os.path.join(outputdir, "periodic_save")
    os.makedirs(savedir)
    periodic_save_callback = ModelCheckpoint(
        dirpath=savedir,
        save_weights_only=True,
        every_n_train_steps=config.periodic_save_interval,
    )

    callbacks.append(periodic_save_callback)

    ## add auroc ckpt saver
    savedir = os.path.join(outputdir, "auroc_save")
    os.makedirs(savedir)
    auroc_save_callback = ModelCheckpoint(
        dirpath=savedir,
        monitor="auroc",
        mode="max",
        save_top_k=2,
        save_weights_only=True,
        filename="{epoch}-{step}-{auroc:.3f}",
    )
    callbacks.append(auroc_save_callback)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        val_check_interval=config.val_check_interval,
        default_root_dir=outputdir,
        callbacks=callbacks,
    )

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
