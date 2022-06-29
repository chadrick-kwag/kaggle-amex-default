import glob, os


def gather_files_from_dir(d):

    return glob.glob(os.path.join(d, "*"))


def gather_files_from_dir_list(d_list):

    files = []

    for d in d_list:
        _files = gather_files_from_dir(d)
        files.extend(_files)

    return files
