import torch, yaml, os, csv, datetime, sqlite3
from .common import gather_files_from_dir_list


class NA_Flag_Dataset(torch.utils.data.Dataset):
    """
    handles n/a for features.
    utilize all features.

    """

    def __init__(self, dir_list, parsing_config_list_file, label_sqlite3_file):

        self.dir_list = dir_list
        assert dir_list, "no dirlist"
        self.files = gather_files_from_dir_list(dir_list)

        assert self.files, "no files"

        assert os.path.exists(parsing_config_list_file)

        with open(parsing_config_list_file, "r") as fd:
            data = yaml.load(fd, Loader=yaml.FullLoader)

        self.parsing_config_list = data["col_parsing_config_list"]

        self.col_name_order = [a["col_name"] for a in self.parsing_config_list]
        self.col_name_to_list_index_dict = {}
        self.col_name_to_type_dict = {}

        for i, a in enumerate(self.parsing_config_list):
            col_name = a["col_name"]
            self.col_name_to_list_index_dict[col_name] = i
            self.col_name_to_type_dict[col_name] = a["type"]

        # load csv label file
        self.label_sqlite3_file = label_sqlite3_file
        if label_sqlite3_file:
            assert os.path.exists(label_sqlite3_file)
            self.con = sqlite3.connect(label_sqlite3_file)
            self.cur = self.con.cursor()
        else:
            self.con = None
            self.cur = None

    def __len__(self):
        return len(self.files)

    @classmethod
    def convert_date_to_float(cls, item_list, date_col_name=None):

        if date_col_name is None:
            date_col_name = "S_2"

        date_values = [a[date_col_name] for a in item_list]

        datetime_obj_list = []

        for d in date_values:
            el_list = d["value"].split("-")
            assert len(el_list) == 3, f"invalid split for date string: {d}"
            n = datetime.datetime(*map(int, el_list))
            datetime_obj_list.append(n)

        min_date = min(datetime_obj_list)

        diff_days_list = []

        for d in datetime_obj_list:
            days = (d - min_date).days
            diff_days_list.append(days)

        for item, days in zip(item_list, diff_days_list):
            item[date_col_name]["value"] = days

        return item_list

    @classmethod
    def get_data_from_single_file(
        cls, f, col_name_to_list_index_dict, parsing_config_list, cur
    ):

        item_list = []
        with open(f, "r") as fd:
            reader = csv.reader(fd)
            header = next(reader)
            valid_header = header[1:]
            cid = None
            for row in reader:
                item = {}
                if cid is None:
                    cid = row[0]
                for elem, h in zip(row[1:], valid_header):
                    parse_config_index = col_name_to_list_index_dict[h]
                    parsing_config = parsing_config_list[parse_config_index]

                    _type = parsing_config["type"]
                    if _type == "date":
                        assert elem != "", "empty value for date"
                        v = elem
                        na = False
                    elif _type == "float":
                        if elem == "":
                            na = True
                            v = 0
                        else:
                            na = False
                            v = float(elem)
                    elif _type == "category":
                        _size = parsing_config["size"]
                        vec = [0] * _size
                        if elem == "":
                            na = True
                        else:
                            na = False
                            idx = parsing_config["mapping"][elem]
                            vec[idx] = 1

                        v = vec

                    if na is True:
                        na = 1
                    else:
                        na = 0

                    item[h] = {"na": na, "value": v}

                item_list.append(item)

        # handle date values among items
        item_list = cls.convert_date_to_float(item_list)

        data = {"item_list": item_list}

        if cur:
            res = cur.execute("SELECT broke from table1 where cid=?", (cid,))

            label = res.fetchone()[0]

            data["label"] = label

        return data

    def __getitem__(self, idx):

        f = self.files[idx]

        data = self.get_data_from_single_file(
            f, self.col_name_to_list_index_dict, self.parsing_config_list, self.cur
        )

        return data

    def collate(self, data_list):

        max_len = max([len(a["item_list"]) for a in data_list])
        batch_key_padding_mask = torch.ones((len(data_list), max_len), dtype=torch.bool)

        padded_data = {}

        # extract data element keys
        data_keys = list(data_list[0]["item_list"][0].keys())
        for k in data_keys:
            padded_data[k] = {"na": [], "value": []}

        for k in data_keys:
            batch_na_list = []
            batch_value_list = []
            for data in data_list:
                l = len(data["item_list"])
                pad_size = max_len - l
                key_values = [d[k]["value"] for d in data["item_list"]]

                first_key_value = key_values[0]

                if isinstance(first_key_value, list):
                    empty_key_value = [0] * len(first_key_value)
                    padded_key_values = key_values + [empty_key_value] * pad_size
                elif isinstance(first_key_value, int) or isinstance(
                    first_key_value, float
                ):
                    empty_key_value = 0
                    padded_key_values = [[a] for a in key_values]
                    padded_key_values = (
                        padded_key_values + [[empty_key_value]] * pad_size
                    )
                else:
                    raise Exception(
                        f"invalid type for first key value: {type(first_key_value)}"
                    )

                na_list = [0] * l
                na_list = na_list + [1] * pad_size

                batch_na_list.append(na_list)
                batch_value_list.append(padded_key_values)

            batch_na_list = torch.IntTensor(batch_na_list).unsqueeze(-1)

            batch_value_list = torch.FloatTensor(batch_value_list)

            padded_data[k] = {"na": batch_na_list, "value": batch_value_list}

        # false marking for key padding mask
        for i, data in enumerate(data_list):
            seq_len = len(data["item_list"])
            batch_key_padding_mask[i, :seq_len] = False

        collated_data = {
            "data": padded_data,
            "key_padding_mask": batch_key_padding_mask,
        }

        if "label" in data_list[0]:
            batch_label = [d["label"] for d in data_list]

            batch_label = torch.FloatTensor(batch_label).unsqueeze(-1)

            # add epsilon to 0 or 1
            batch_label = torch.maximum(
                batch_label, torch.ones_like(batch_label) * 1e-8
            )
            batch_label = torch.minimum(
                batch_label, torch.ones_like(batch_label) - 1e-8
            )

            collated_data["label"] = batch_label

        return collated_data
