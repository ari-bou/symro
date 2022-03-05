import numpy as np
from typing import Dict

import symro.util.util as util


def to_alamo(
    x: np.ndarray,  # (m, n_x)
    y: np.ndarray,  # (m, n_y)
    x_min: np.ndarray,  # (n_x)
    x_max: np.ndarray,  # (n_x)
    x_col: np.ndarray,  # (n_x)
    y_col: np.ndarray,  # (n_y)
    valid_split: float = 0,  # [0, 1]
    file_name: str = "data.alm",
    dir_path: str = None,
    options: Dict[str, str] = None,
):

    data = np.concatenate((x, y), axis=1)

    # process input bounds

    x_max = np.where(x_min == x_max, x_max + 0.1, x_max)

    x_min = [str(x) for x in x_min]
    x_max = [str(x) for x in x_max]

    # validation split
    if valid_split is not None and valid_split > 0:

        split_pos = int(np.floor((1 - valid_split) * data.shape[0]))

        data_train = data[:split_pos, :]
        data_valid = data[split_pos:, :]

        nvalsets = 1
        nvaldata = data_valid.shape[0]

    # no validation split
    else:

        data_train = data
        data_valid = None

        nvalsets = 0
        nvaldata = 0

    # generate alm file

    lines = [
        "ninputs {0}\n".format(len(x_col)),
        "noutputs {0}\n".format(len(y_col)),
        "xlabels {0}\n".format(" ".join(x_col)),
        "zlabels {0}\n".format(" ".join(y_col)),
        "xmin {0}\n".format(" ".join(x_min)),
        "xmax {0}\n".format(" ".join(x_max)),
        "ndata {0}\n".format(data_train.shape[0]),
        "nvalsets {0}\n".format(nvalsets),
        "nvaldata {0}\n".format(nvaldata),
        "solvemip 1\n",
        "PRINT_TO_FILE 1\n",
        "FUNFORM 5\n",
        "GAMSSOLVER BARON\n",
    ]

    if options is not None:
        for option_sym, option_val in options.items():
            option_val = option_val.replace("'", "").replace('"', "")
            lines.append("{0} {1}\n".format(option_sym, option_val))

    lines.append("BEGIN_DATA\n")

    for row in data_train:
        row = [str(v) for v in row]
        lines.append(" ".join(row) + " \n")

    lines.append("END_DATA\n")

    if data_valid is not None:

        lines.append("BEGIN_VALDATA\n")

        for row in data_valid:
            row = [str(v) for v in row]
            lines.append(" ".join(row) + " \n")

        lines.append("END_VALDATA\n")

    # write to file
    util.write_file(dir_path=dir_path, file_name=file_name, text="".join(lines))
