import csv, json, datetime, os
from pprint import pprint


def create_outputdir():

    bn = os.path.basename(__file__)
    fn, _ = os.path.splitext(bn)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    outputdir = f"testoutput/{fn}/{timestamp}"

    os.makedirs(outputdir)

    return outputdir


def main(pred_csv_file, gt_csv_file, outputdir):

    assert os.path.exists(pred_csv_file)
    assert os.path.exists(gt_csv_file)

    gt_dict = {}
    with open(gt_csv_file, "r") as fd:
        reader = csv.reader(fd)

        next(reader)

        for r in reader:
            cid, gt = r

            gt_dict[cid] = int(gt)

    print("gt load done")

    total_count = 0
    match_count = 0

    with open(pred_csv_file, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")

        next(reader)

        for r in reader:
            total_count += 1
            cid, pred = r
            pred = int(pred)

            gt = gt_dict[cid]

            if gt == pred:
                match = True
                match_count += 1
            else:
                match = False

    acc = match_count / total_count

    result = {"total count": total_count, "match count": match_count, "acc": acc}

    pprint(result)

    if outputdir is not None:
        p = os.path.join(outputdir, "result.json")

        with open(p, "w") as fd:
            json.dump(result, fd, indent=4, ensure_ascii=False)

    return


if __name__ == "__main__":

    gt_csv_file = "/home/chadrick/prj/kaggle/amex_default/data/train_labels.csv"
    pred_csv_file = "/home/chadrick/prj/kaggle/amex_default/infer/testoutput/infer/220703_160448/output/result.csv"

    outputdir = create_outputdir()

    main(pred_csv_file, gt_csv_file, outputdir)
