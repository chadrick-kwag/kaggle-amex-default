import csv, datetime, os, json
from pprint import pprint


def main(infer_result_csv, threshold, outputdir):

    assert os.path.exists(infer_result_csv)

    p = os.path.join(outputdir, "submit.csv")

    writer_fd = open(p, "w")

    writer = csv.writer(writer_fd)
    writer.writerow(["customer_ID", "prediction"])

    with open(infer_result_csv, "r") as fd:
        reader = csv.reader(fd)

        next(reader)
        i = 0
        for r in reader:
            i += 1

            if i % 1000 == 0:
                print(f"progress : {i}")
            cid, prob = r

            prob = float(prob)

            if prob >= threshold:
                res = 1
            else:
                res = 0

            writer.writerow([cid, res])

    writer_fd.close()

    info = {
        "infer result csv": infer_result_csv,
        "theshold": threshold.item(),
        "row count": i,
    }

    pprint(info)

    p = os.path.join(outputdir, "result.json")

    with open(p, "w") as fd:
        json.dump(info, fd, indent=4, ensure_ascii=False)

    return


def create_outputdir():

    bn = os.path.basename(__file__)
    fn, _ = os.path.splitext(bn)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f"testoutput/{fn}/{timestamp}"
    os.makedirs(outputdir)

    return outputdir


if __name__ == "__main__":

    infer_result_csv = "/home/chadrick/prj/kaggle/kaggle-amex-default/infer/testoutput/infer_v2/220807_191405/out_prob.csv"

    outputdir = create_outputdir()

    threshold = 0.255

    main(infer_result_csv, threshold, outputdir)
