""" 
compared to `infer.py`, this directly builds model without pl model wrapper
and loades only model state dict from the checkpoint file
"""


import os, sys, torch, datetime, argparse, yaml, csv
from tqdm import tqdm
from munch import munchify

if __name__ == "__main__":
    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    if rootdir not in sys.path:
        sys.path.append(rootdir)

from model.model import Model_v2
from dataset.na_flag_dataset import NA_Flag_Dataset


def get_fn(f):
    bn = os.path.basename(f)
    return os.path.splitext(bn)[0]


def recursive_send_to_device(data, device):

    if isinstance(data, dict):
        for k, v in data.items():
            _v = recursive_send_to_device(v, device)
            data[k] = _v
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


class Inference:
    def __init__(self, config):

        self.config = config

        self.setup_outputdir()
        self.save_used_config()
        self.setup_device()
        self.setup_model()
        self.setup_dataloader()
        self.setup_prob_output_file_desc_and_csv_writer()

    def setup_device(self):
        print("setup device ...")

        value = self.config.device

        if value == "cpu":
            self.device = torch.device("cpu")
        elif isinstance(value, int):
            self.device = torch.device(f"cuda:{value}")

        else:
            raise Exception(f"invalid device: {value}")

    def setup_outputdir(self):
        print("setup outputdir...")

        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

        outputdir = f"testoutput/infer_v2/{timestamp}"

        os.makedirs(outputdir)

        self.outputdir = outputdir

    def save_used_config(self):
        print("save used config...")

        p = os.path.join(self.outputdir, "usedconfig.yaml")

        with open(p, "w") as fd:
            yaml.dump(self.config, fd)

    def setup_prob_output_file_desc_and_csv_writer(self):

        p = os.path.join(self.outputdir, "out_prob.csv")

        self.fd = open(p, "w")

        self.writer = csv.writer(self.fd)

        self.writer.writerow(["cid", "prob"])

    def setup_model(self):
        print("setup model ...")

        assert os.path.exists(self.config.ckpt)

        _type = self.config.model.type

        if _type == "Model_v2":
            _cls = Model_v2
        else:
            raise Exception(f"invalid model type: {_type}")

        args = self.config.model.args
        # handle `embedding_config_list`
        if hasattr(args, "embedding_config_list"):
            assert os.path.exists(
                args.embedding_config_list
            ), f"embedding_config_list file not exist"

            with open(args.embedding_config_list, "r") as fd:
                data = yaml.load(fd, Loader=yaml.FullLoader)
                embedding_config_list = data["model_embedding_config_list"]
            args.embedding_config_list = embedding_config_list

        args = vars(args)
        self.model = _cls(**args)

        # load from ckpt
        ckpt_data = torch.load(self.config.ckpt)
        state_dict = ckpt_data["state_dict"]

        name_modified_state_dict = {}

        for k, v in state_dict.items():
            if k[:6] == "model.":
                _k = k[6:]
                name_modified_state_dict[_k] = v

        self.model.load_state_dict(name_modified_state_dict)

        self.model.to(self.device)

    def setup_dataloader(self):
        print("setup dataloader...")

        _type = self.config.infer_data.dataset.type

        if _type == "NA_Flag_Dataset":
            _cls = NA_Flag_Dataset
        else:
            raise Exception(f"invalide datset type: {_type}")

        args = vars(self.config.infer_data.dataset.args)
        dataset = _cls(**args)

        _type = self.config.infer_data.dataloader.type

        if _type == "default":
            _cls = torch.utils.data.DataLoader
        else:
            raise Exception(f"invalid dataloader type: {_type}")

        args = vars(self.config.infer_data.dataloader.args)
        args["collate_fn"] = dataset.collate
        dataloader = _cls(dataset, **args)

        self.dataloader = dataloader

    def _get_pred_prob_from_model_output(self, output):

        probs = output[:, 0, 1]

        return probs.tolist()

    def run(self):

        self.model.eval()

        for data in tqdm(self.dataloader):

            input_data_keys = ["key_padding_mask", "data"]

            input_data = {}
            for k in input_data_keys:
                v = data[k]
                input_data[k] = recursive_send_to_device(v, self.device)

            with torch.no_grad():
                output = self.model(**input_data)

            clsf_output = output["clsf_output"].detach()
            prob_tensor = torch.softmax(clsf_output, dim=-1)

            probs = self._get_pred_prob_from_model_output(prob_tensor.cpu())

            files = data["file"]

            for f, p in zip(files, probs):
                fn = get_fn(f)

                self.writer.writerow([fn, p])


def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="config file")
    args = parser.parse_args()

    assert os.path.exists(args.config)

    with open(args.config, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    config = munchify(config)

    inference = Inference(config)

    inference.run()


if __name__ == "__main__":

    cli()
