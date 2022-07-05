import torch, csv, datetime, os, pickle
from .common import gather_files_from_dir_list


class NoNanColsDataset_v1(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_list,
        label_csv_file,
        data_flatten_size,
        missing_default_value_pkl_file=None,
    ):

        self.dir_list = dir_list
        assert dir_list, "no dir list"
        self.files = gather_files_from_dir_list(dir_list)

        assert data_flatten_size > 0, f"invalid data flatten size: {data_flatten_size}"
        self.data_flatten_size = data_flatten_size

        self.missing_default_value_pkl_file = missing_default_value_pkl_file

        self.header_flatten_order_list = [
            "S_2",
            "D_39",
            "B_1",
            "R_1",
            "B_4",
            "B_5",
            "R_2",
            "D_47",
            "B_7",
            "D_51",
            "B_9",
            "R_3",
            "B_10",
            "S_5",
            "B_11",
            "S_6",
            "R_4",
            "B_12",
            "S_8",
            "R_5",
            "D_58",
            "B_14",
            "D_60",
            "S_11",
            "D_63",
            "D_65",
            "B_18",
            "S_12",
            "R_6",
            "S_13",
            "B_21",
            "D_71",
            "S_15",
            "B_23",
            "P_4",
            "D_75",
            "B_24",
            "R_8",
            "S_16",
            "R_10",
            "R_11",
            "S_17",
            "B_28",
            "R_13",
            "R_15",
            "R_16",
            "S_18",
            "D_86",
            "R_17",
            "R_18",
            "B_31",
            "S_19",
            "R_19",
            "B_32",
            "S_20",
            "R_21",
            "R_22",
            "R_23",
            "D_92",
            "D_93",
            "D_94",
            "R_24",
            "R_25",
            "D_96",
            "B_36",
            "D_127",
            "R_28",
        ]
        self.header_index_dict = {
            "S_2": 1,
            "D_39": 3,
            "B_1": 4,
            "R_1": 6,
            "B_4": 13,
            "B_5": 15,
            "R_2": 16,
            "D_47": 18,
            "B_7": 22,
            "D_51": 25,
            "B_9": 26,
            "R_3": 27,
            "B_10": 30,
            "S_5": 32,
            "B_11": 33,
            "S_6": 34,
            "R_4": 36,
            "B_12": 38,
            "S_8": 39,
            "R_5": 43,
            "D_58": 44,
            "B_14": 46,
            "D_60": 48,
            "S_11": 51,
            "D_63": 53,
            "D_65": 55,
            "B_18": 58,
            "S_12": 63,
            "R_6": 64,
            "S_13": 65,
            "B_21": 66,
            "D_71": 70,
            "S_15": 72,
            "B_23": 73,
            "P_4": 75,
            "D_75": 77,
            "B_24": 79,
            "R_8": 86,
            "S_16": 88,
            "R_10": 90,
            "R_11": 91,
            "S_17": 95,
            "B_28": 97,
            "R_13": 98,
            "R_15": 101,
            "R_16": 103,
            "S_18": 106,
            "D_86": 107,
            "R_17": 109,
            "R_18": 110,
            "B_31": 112,
            "S_19": 113,
            "R_19": 114,
            "B_32": 115,
            "S_20": 116,
            "R_21": 118,
            "R_22": 121,
            "R_23": 122,
            "D_92": 124,
            "D_93": 125,
            "D_94": 126,
            "R_24": 127,
            "R_25": 128,
            "D_96": 129,
            "B_36": 141,
            "D_127": 168,
            "R_28": 177,
        }

        self.header_type_dict = {
            "S_2": {"type": "date"},
            "D_39": {"type": "float"},
            "B_1": {"type": "float"},
            "R_1": {"type": "float"},
            "B_4": {"type": "float"},
            "B_5": {"type": "float"},
            "R_2": {"type": "float"},
            "D_47": {"type": "float"},
            "B_7": {"type": "float"},
            "D_51": {"type": "float"},
            "B_9": {"type": "float"},
            "R_3": {"type": "float"},
            "B_10": {"type": "float"},
            "S_5": {"type": "float"},
            "B_11": {"type": "float"},
            "S_6": {"type": "float"},
            "R_4": {"type": "float"},
            "B_12": {"type": "float"},
            "S_8": {"type": "float"},
            "R_5": {"type": "float"},
            "D_58": {"type": "float"},
            "B_14": {"type": "float"},
            "D_60": {"type": "float"},
            "S_11": {"type": "float"},
            "D_63": {
                "type": "category",
                "mapping": {"XZ": 0, "XM": 1, "XL": 2, "CL": 3, "CR": 4, "CO": 5},
            },
            "D_65": {"type": "float"},
            "B_18": {"type": "float"},
            "S_12": {"type": "float"},
            "R_6": {"type": "float"},
            "S_13": {"type": "float"},
            "B_21": {"type": "float"},
            "D_71": {"type": "float"},
            "S_15": {"type": "float"},
            "B_23": {"type": "float"},
            "P_4": {"type": "float"},
            "D_75": {"type": "float"},
            "B_24": {"type": "float"},
            "R_8": {"type": "float"},
            "S_16": {"type": "float"},
            "R_10": {"type": "float"},
            "R_11": {"type": "float"},
            "S_17": {"type": "float"},
            "B_28": {"type": "float"},
            "R_13": {"type": "float"},
            "R_15": {"type": "float"},
            "R_16": {"type": "float"},
            "S_18": {"type": "float"},
            "D_86": {"type": "float"},
            "R_17": {"type": "float"},
            "R_18": {"type": "float"},
            "B_31": {"type": "float"},
            "S_19": {"type": "float"},
            "R_19": {"type": "float"},
            "B_32": {"type": "float"},
            "S_20": {"type": "float"},
            "R_21": {"type": "float"},
            "R_22": {"type": "float"},
            "R_23": {"type": "float"},
            "D_92": {"type": "float"},
            "D_93": {"type": "float"},
            "D_94": {"type": "float"},
            "R_24": {"type": "float"},
            "R_25": {"type": "float"},
            "D_96": {"type": "float"},
            "B_36": {"type": "float"},
            "D_127": {"type": "float"},
            "R_28": {"type": "float"},
        }

        self.label_csv_file = label_csv_file
        self.load_label()
        self.load_missing_default_value_pkl_file()

    def load_missing_default_value_pkl_file(self):

        self.missing_default_value_dict = None

        if self.missing_default_value_pkl_file is None:
            return

        with open(self.missing_default_value_pkl_file, "rb") as fd:

            data = pickle.load(fd)

        self.missing_default_value_dict = data

    def load_label(self):

        # if label csv file is None, this dataset is being used for inference. no need to prepare gt
        if self.label_csv_file is None:
            self.label_dict = None
            return

        assert os.path.exists(self.label_csv_file)
        self.label_dict = {}
        with open(self.label_csv_file, "r") as fd:
            reader = csv.reader(fd)

            header = next(reader)

            for r in reader:
                cid = r[0]
                label = float(r[1])

                self.label_dict[cid] = label

    def __len__(self):

        return len(self.files)

    def convert_string_to_appropriate_value(self, s, type_info):
        _type = type_info["type"]

        if _type == "float":
            return float(s)
        elif _type == "category":
            return type_info["mapping"][s]
        elif _type == "date":
            el_list = s.split("-")
            assert len(el_list) == 3, f"invalid date format: {s}"

            d = datetime.datetime(*map(int, el_list))
            return d

    def convert_date_values_from_datetime_to_int_value(self, data_list):

        min_date = min([d["S_2"] for d in data_list])

        for d in data_list:
            date = d["S_2"]
            days = (date - min_date).days
            d["S_2"] = days

        return

    def get_filtered_dict_data_from_row(self, row):

        data = {}

        for k, i in self.header_index_dict.items():
            v = row[i]

            # if value is missing, fill it with default value if missing default setup exists
            if self.missing_default_value_dict is not None and v == "":
                default_value = self.missing_default_value_dict[k]
                v = default_value

            t = self.header_type_dict[k]

            real_v = self.convert_string_to_appropriate_value(v, t)

            data[k] = real_v

        return data

    def convert_row_data_dict_to_vector(self, row_data):

        val_list = []

        for name in self.header_flatten_order_list:
            v = row_data[name]
            t = self.header_type_dict[name]

            if t["type"] == "category":
                full_size = len(t["mapping"])
                arr = [0] * full_size
                arr[v] = 1
                val_list.extend(arr)
            else:
                val_list.append(float(v))

        return val_list

    def get_data_from_file(self, f):

        with open(f, "r") as fd:
            reader = csv.reader(fd)

            next(reader)

            lines = list(reader)

        assert lines, "no rows found"

        # get cid
        cid = lines[0][0]

        if self.label_dict:
            label = self.label_dict[cid]
        else:
            label = None
        # handle lines

        data_list = []
        for l in lines:
            data = self.get_filtered_dict_data_from_row(l)
            data_list.append(data)

        self.convert_date_values_from_datetime_to_int_value(data_list)

        # convert each data to vector
        type_token_list = [1] + [0] * len(data_list)

        data_arr_list = [[0] * self.data_flatten_size] + [
            self.convert_row_data_dict_to_vector(d) for d in data_list
        ]

        return_data = {
            "file": f,
            "token_types": torch.IntTensor(type_token_list),
            "token_features": torch.FloatTensor(data_arr_list),
        }

        if label:
            return_data["label"] = torch.FloatTensor([label])

        return return_data

    @classmethod
    def collate(cls, data_list):

        # check if label exist in data
        label_exist = False
        if hasattr(data_list[0], "label"):
            label_exist = True

        batch_files = []
        batch_token_type = []
        batch_token_features = []
        batch_label = []
        seq_len_list = []

        for d in data_list:
            batch_files.append(d["file"])
            seq_len_list.append(d["token_types"].shape[0])
            batch_token_type.append(d["token_types"])
            batch_token_features.append(d["token_features"])

            if label_exist:
                batch_label.append(d["label"])

        max_seq_len = max(seq_len_list)

        token_types_dtype = data_list[0]["token_types"].dtype
        batch_token_types = torch.zeros(
            (len(data_list), max_seq_len), dtype=token_types_dtype
        )
        for i, d in enumerate(data_list):
            seq_len = d["token_types"].shape[0]
            batch_token_types[i, :seq_len] = d["token_types"]

        features_dim = data_list[0]["token_features"].shape[1]
        features_dtype = data_list[0]["token_features"].dtype

        batch_token_features = torch.zeros(
            (len(data_list), max_seq_len, features_dim), dtype=features_dtype
        )
        for i, d in enumerate(data_list):
            seq_len = d["token_features"].shape[0]
            batch_token_features[i, :seq_len] = d["token_features"]

        if label_exist:
            label_dim = data_list[0]["label"].shape[0]
            label_dtype = data_list[0]["label"].dtype
            batch_label = torch.zeros((len(data_list), label_dim), dtype=label_dtype)

            for i, d in enumerate(data_list):
                batch_label[i] = d["label"]

        # create key_padding_mask
        batch_key_padding_mask = torch.ones(
            (len(data_list), max_seq_len), dtype=torch.bool
        )
        # true will ignore
        for i, l in enumerate(seq_len_list):
            batch_key_padding_mask[i, :l] = False

        batched_data = {
            "files": batch_files,
            "token_types": batch_token_types,
            "token_features": batch_token_features,
            "key_padding_mask": batch_key_padding_mask,
        }

        if label_exist:
            batched_data["label"]: batch_label

        return batched_data

    def __getitem__(self, idx):

        f = self.files[idx]

        return self.get_data_from_file(f)
