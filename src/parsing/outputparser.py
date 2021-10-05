from typing import Tuple


def parse_conopt_output(literal: str) -> Tuple[bool, float, int]:

    is_feasible = False
    v_opt = 0
    last_fbl_iter = 0

    lines = literal.split("\r\n")

    can_parse_line = False
    for line in lines:

        line = line.strip()
        tokens = line.split()

        if not can_parse_line:
            if len(tokens) >= 2:
                if tokens[0] == "iter" and tokens[1] == "phase":
                    can_parse_line = True
        else:

            if len(tokens) == 0:
                break

            else:
                iter = int(tokens[0])
                suminf = float(tokens[3])
                v = float(tokens[5])

                if suminf == 0:
                    is_feasible = True
                    v_opt = v
                    last_fbl_iter = iter

    return is_feasible, v_opt, last_fbl_iter
