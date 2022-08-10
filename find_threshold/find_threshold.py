from torchmetrics import ROC
import csv, os, datetime, sqlite3, numpy as np, json, matplotlib.pyplot as plt, pickle, torch


def main(csv_file, label_sqlite_db_file, outputdir):

    assert os.path.exists(csv_file)

    assert os.path.exists(label_sqlite_db_file)

    con = sqlite3.connect(label_sqlite_db_file)
    cur = con.cursor()

    def get_label_of_cid(cid):

        result = cur.execute("select broke from table1 where cid=?", (cid,))

        label = result.fetchone()[0]

        return label

    roc = ROC()

    print("loading csv file predictions...")
    with open(csv_file, "r") as fd:
        reader = csv.reader(fd)
        next(reader)
        pred = []
        target = []
        i = 0
        for r in reader:
            i += 1
            if i % 5000 == 0:
                print(f"csv read progress: {i}")
            cid, prob = r
            prob = float(prob)
            label = get_label_of_cid(cid)

            pred.append(prob)
            target.append(label)

    # save pred and target as pickle
    data = {"pred": pred, "target": target}

    p = os.path.join(outputdir, "pred_target.pkl")

    with open(p, "wb") as fd:
        pickle.dump(data, fd)

    print("calculate roc...")
    pred_tensor = torch.FloatTensor(pred)
    target_tensor = torch.FloatTensor(target)
    fpr, tpr, thresholds = roc(pred_tensor, target_tensor)
    fpr_arr = np.array(fpr)
    tpr_arr = np.array(tpr)

    gmean_arr = np.sqrt(tpr_arr * (1 - fpr_arr))

    max_gmean_index = np.argmax(gmean_arr)

    opt_threshold = thresholds[max_gmean_index]

    result = {
        "csv file": csv_file,
        "labe sqlite db": label_sqlite_db_file,
        "opt threshold": opt_threshold.item(),
    }
    p = os.path.join(outputdir, "result.json")

    with open(p, "w") as fd:
        json.dump(result, fd, indent=4, ensure_ascii=False)

    return opt_threshold


def create_outputdir():

    bn = os.path.basename(__file__)
    fn, _ = os.path.splitext(bn)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f"testoutput/{fn}/{timestamp}"
    os.makedirs(outputdir)

    return outputdir


if __name__ == "__main__":

    outputdir = create_outputdir()

    label_sqlite_db = "/home/chadrick/prj/kaggle/kaggle-amex-default/teststuff/sqlite/testoutput/t1/220804_194730/label.db"

    infer_result_csv = "/home/chadrick/prj/kaggle/kaggle-amex-default/infer/testoutput/infer_v2/220808_115039_valid_infer/out_prob.csv"

    main(infer_result_csv, label_sqlite_db, outputdir)
