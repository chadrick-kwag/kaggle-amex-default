import os, sys, datetime, yaml, argparse, json
from munch import munchify

if __name__ == "__main__":
    rootdir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    if rootdir not in sys.path:
        sys.path.append(rootdir)

from infer.infer_v2 import Inference
from find_threshold.find_threshold import main as find_threshold_main
from submit.convert_infer_result_to_submit_file import main as submit_convert_main


def run_test_infer(config, outputdir):

    test_infer_outputdir = os.path.join(outputdir, "test_infer")
    os.makedirs(test_infer_outputdir)

    infer_config = config.test_infer_config

    infer_config.outputdir = test_infer_outputdir

    inference = Inference(infer_config)

    inference.run()

    return test_infer_outputdir


def run_valid_infer(config, outputdir):

    valid_infer_outputdir = os.path.join(outputdir, "valid_infer")
    os.makedirs(valid_infer_outputdir)

    infer_config = config.valid_infer_config

    infer_config.outputdir = valid_infer_outputdir

    inference = Inference(infer_config)

    inference.run()

    return valid_infer_outputdir


def find_threshold(config, valid_outputdir, outputdir):

    valid_result_csv = os.path.join(valid_outputdir, "out_prob.csv")
    assert os.path.exists(valid_result_csv)

    _outputdir = os.path.join(outputdir, "find_threshold")
    os.makedirs(_outputdir)
    threshold = find_threshold_main(
        valid_result_csv, config.label_sqlite_db_file, _outputdir
    )

    return threshold


def do_submit_conversion(infer_result_csv, threshold, outputdir):

    submit_convert_main(infer_result_csv, threshold, outputdir)


def save_used_config_yaml(config, outputdir):

    # save used config
    p = os.path.join(outputdir, "usedconfig.yaml")

    with open(p, "w") as fd:
        yaml.dump(config, fd)


def cont_from_valid_infer(config, outputdir):

    test_output = config.cont_dir
    assert os.path.exists(test_output)


def submit_stage(outputdir, test_output, threshold):

    submit_outputdir = os.path.join(outputdir, "submit")
    os.makedirs(submit_outputdir)

    test_result_csv = os.path.join(test_output, "out_prob.csv")
    assert os.path.exists(test_result_csv)

    print("creating submit csv...")
    do_submit_conversion(test_result_csv, threshold, submit_outputdir)


def full_flow(config, outputdir):

    print("running test infer...")
    test_output = run_test_infer(config, outputdir)

    print("running valid infer...")
    valid_output = run_valid_infer(config, outputdir)

    print("calculating threshold...")
    threshold = find_threshold(config, valid_output, outputdir)

    submit_stage(outputdir, test_output, threshold)


def save_result_json(outputdir):

    info = {"finish time": datetime.datetime.now().isoformat()}
    p = os.path.join(outputdir, "result.json")

    with open(p, "w") as fd:
        json.dump(info, fd, indent=4, ensure_ascii=False)


def main(config, outputdir):

    save_used_config_yaml(config, outputdir)

    full_flow(config, outputdir)

    save_result_json(outputdir)
    return


def create_outputdir():

    bn = os.path.basename(__file__)
    fn, _ = os.path.splitext(bn)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f"testoutput/{fn}/{timestamp}"
    os.makedirs(outputdir)

    return outputdir


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="config file")

    args = parser.parse_args()

    assert os.path.exists(args.config)

    with open(args.config, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    config = munchify(config)

    outputdir = create_outputdir()

    main(config, outputdir)


if __name__ == "__main__":
    cli()
