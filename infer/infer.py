import sys, os, argparse, yaml, pytorch_lightning as pl, torch, datetime, csv
from venv import create
from munch import munchify
from tqdm import tqdm

if __name__ == "__main__":
    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    if rootdir not in sys.path:
        sys.path.append(rootdir)

from train.trainer import Model_v1
from dataset.no_nan_dataset import NoNanColsDataset_v1


def get_fn(f):
    bn = os.path.basename(f)
    return os.path.splitext(bn)[0]


def create_outputdir():

    bn = os.path.basename(__file__)
    fn, _ = os.path.splitext(bn)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    outputdir = f"testoutput/{fn}/{timestamp}"

    os.makedirs(outputdir)

    return outputdir


def main(config, outputdir):

    assert os.path.exists(config.ckpt), "ckpt not exist"

    model = Model_v1(
        config=config,
        d_model=config.model.d_model,
        nhead=config.model.n_head,
        activation=config.model.activation,
        num_layers=config.model.num_layers,
        token_type_embedding_size=config.model.token_type_embedding_size,
        feature_dim=config.model.feature_dim,
    )

    model = model.load_from_checkpoint(
        config.ckpt,
        config=config,
        d_model=config.model.d_model,
        nhead=config.model.n_head,
        activation=config.model.activation,
        num_layers=config.model.num_layers,
        token_type_embedding_size=config.model.token_type_embedding_size,
        feature_dim=config.model.feature_dim,
    )

    print("model load ckpt complete")

    dataset = NoNanColsDataset_v1(
        config.infer_data.dir_list,
        None,
        config.infer_data.data_flatten_size,
        missing_default_value_pkl_file=config.infer_data.missing_default_value_pkl_file
        if hasattr(config.infer_data, "missing_default_value_pkl_file")
        else None,
    )

    print("dataset load complete")

    infer_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=NoNanColsDataset_v1.collate,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )

    print("dataloader load complete")

    device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    # prepare outputdir

    savedir = os.path.join(outputdir, "output")
    os.makedirs(savedir)

    # save config
    p = os.path.join(outputdir, "usedconfig.yaml")

    with open(p, "w") as fd:
        yaml.dump(config, fd)

    p = os.path.join(savedir, "result.csv")
    with open(p, "w") as fd:
        writer = csv.writer(fd, delimiter="\t")

        writer.writerow(["customer_ID", "prediction", "raw_prediction"])

        for data in tqdm(infer_dataloader):
            input_data = {}

            input_data["token_types"] = data["token_types"].to(device)
            input_data["token_features"] = data["token_features"].to(device)
            input_data["src_key_padding_mask"] = data["key_padding_mask"].to(device)

            with torch.no_grad():
                output = model(**input_data)

                cls_output = output["clsf_output"][:, 0]
                cls_output = torch.softmax(cls_output, -1)

                default_prob = cls_output[:, None, 1]

                default_threshold_result = default_prob > config.threshold
                default_threshold_result = default_threshold_result.int().cpu()

                for f, r, p in zip(
                    data["files"], default_threshold_result, default_prob.cpu()
                ):
                    fn = get_fn(f)
                    writer.writerow((fn, r.item(), p.item()))

    print("done")


def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="config file path")

    args = parser.parse_args()

    assert os.path.exists(args.config), "config file not exist"

    with open(args.config, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    config = munchify(config)

    outputdir = create_outputdir()

    main(config, outputdir)


if __name__ == "__main__":

    cli()
