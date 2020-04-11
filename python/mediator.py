# required formats  (pay attention to spaces and @ !)
# 1. prompt side input with "@ [b]lack / [w]hite ?"
# 2. receive side as "b" / "w"
# 3. prompt action input with "@ action ?"
# 4. output action with "@ action : \w+"
# 5. output result with "@ result : black=[0-9]+ white=[0-9]+"
# 6. pass is denoted as "pass" and treated like other actions
# 7. all prompts and outputs must be followed by "\n"
# other prompts and outputs are ignored

import argparse
import subprocess
import re
import glob
import os
import numpy as np


def flip_side(side):
    if side == 'b':
        return 'w'
    else:
        return 'b'


def read_until(proc, phrase):
    # print(f"read_until {phrase}")
    contents = ""
    while True:
        line = proc.stdout.readline()
        # print(f"read_until received: \"{line}\"")
        if len(line) == 0:
            break

        contents += line

        if phrase in contents:
            break

    return contents


def play_game(cmd1, cmd2, side1, log_file, quiet):
    if side1 == 'b':
        idx2side = {1: 'b', 2: 'w'}
        side2idx = {'b': 1, 'w': 2}
    elif side1 == 'w':
        idx2side = {1: 'w', 2: 'b'}
        side2idx = {'b': 2, 'w': 1}
    else:
        raise RuntimeError(f"side ({side}) is invalid")

    proc1 = subprocess.Popen(cmd1.strip().split(" "), encoding='UTF-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(cmd2.strip().split(" "), encoding='UTF-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    idx2proc = {1: proc1, 2: proc2}

    # new line is necessary for any prompt (otherwise buffered)
    SIDE_PROMPT = "@ [b]lack / [w]hite ?"
    ACTION_PROMPT = "@ action ?"
    ACTION_OUTPUT = "@ action :"
    RESULT_OUTPUT = "@ result :"

    for i in [1, 2]:
        # read until side prompt
        contents = read_until(idx2proc[i], SIDE_PROMPT)
        # print(f"proc[{i}] contents = \"{contents}\"")
        # send *player* side (not computer side!)
        idx2proc[i].stdin.write(flip_side(idx2side[i]) + "\n")
        idx2proc[i].stdin.flush()

    # start game
    side = 'b'
    count = 0

    while True:
        idx_send, idx_recv = side2idx[side], side2idx[flip_side(side)]

        # receive action
        contents = read_until(idx2proc[idx_send], ACTION_OUTPUT)
        # print(f"received: \"{contents}\"")
        m_action = re.search(rf"{ACTION_OUTPUT}\s*(\w+)", contents)

        if m_action:
            action = m_action.group(1)
            if not quiet:
                print(f"{count}: side={side} action=\"{action}\"")
            log_file.write(action + "\n")
        else:
            # action not received => 2 cases
            # 1) result is received (from both programs)  2) error
            idx2result = {}

            m_result = re.search(rf"{RESULT_OUTPUT}\s*black\s*=\s*([0-9]+)\s+white\s*=\s*([0-9]+)", contents)
            if m_result:
                idx2result[idx_send] = (int(m_result.group(1)), int(m_result.group(2)))
            else:
                print(f"disconnected unexpectedly by program{idx_send}")
                exit(-1)

            contents = read_until(idx2proc[idx_recv], RESULT_OUTPUT)
            m_result = re.search(rf"{RESULT_OUTPUT}\s*black\s*=\s*([0-9]+)\s+white\s*=\s*([0-9]+)", contents)
            if m_result:
                idx2result[idx_recv] = (int(m_result.group(1)), int(m_result.group(2)))
            else:
                print(f"disconnected unexpectedly by program{idx_recv}")
                exit(-1)

            break

        # send action
        contents = read_until(idx2proc[idx_recv], ACTION_PROMPT)
        # print(f"not in turn contents = \"{contents}\"")
        idx2proc[idx_recv].stdin.write(action + "\n")
        idx2proc[idx_recv].stdin.flush()
        # print(f"send action \"{action}\"")

        side = flip_side(side)
        count += 1


    for i in [1, 2]:
        result_str = f"result from program{i}: black={idx2result[i][0]} white={idx2result[i][1]}"
        print(result_str)
        log_file.write(result_str)

    if idx2result[1] != idx2result[2]:
        print(f"inconsistent results!")
        exit(-1)

    proc1.wait()
    proc2.wait()
    return idx2result[1]


def get_stats():
    file = open("log1.txt", "w")
    cmd1 = "../cpp/build/play 1 100 --n 100 --d 1"

    side1 = 'b'
    for i in range(10, 100, 10):
        print(f"side1 = {side1}  i = {i}")
        cmd2 = f"../cpp/build/play 1 {i} --n 100 --d 1"
        result = play_game(cmd1, cmd2, side1, file, quiet=True)
        print(f"result: black={result[0]} white={result[1]}")
    print()
    side1 = 'w'
    for i in range(10, 100, 10):
        print(f"side1 = {side1}  i = {i}")
        cmd2 = f"../cpp/build/play 1 {i} --n 100 --d 1"
        result = play_game(cmd1, cmd2, side1, file, quiet=True)
        print(f"result: black={result[0]} white={result[1]}")

    file.close()


if __name__ == '__main__':
    # get_stats()
    # exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd1", type=str, help="command for the first program")
    parser.add_argument("cmd2", type=str, help="command for the second program")
    parser.add_argument("-n", "--n-games", type=int, default=1, help="number of games to play")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--side1", type=str)
    args = parser.parse_args()

    if not args.side1:  # side1 not specified in options
        print("side ([b]lack / [w]hite) of the first program ? ", end="")
        side1 = input()
    else:
        side1 = args.side1

    log_fname = f"log.txt"
    with open(log_fname, "w") as file:
        results = []
        for i in range(args.n_games):
            print(f"game {i+1}")
            results.append(play_game(args.cmd1, args.cmd2, side1, file, args.quiet))

    results = np.array(results, dtype=np.float)

    win_count_b = (results[:, 0] > results[:, 1]).sum()
    win_count_w = (results[:, 0] < results[:, 1]).sum()
    draw_count = (results[:, 0] == results[:, 1]).sum()
    win_rate_b = win_count_b / args.n_games * 100
    win_rate_w = win_count_w / args.n_games * 100
    draw_rate = draw_count / args.n_games * 100
    print(f"black win rate = {win_rate_b:.2f} %  ({win_count_b} / {args.n_games})")
    print(f"white win rate = {win_rate_w:.2f} %  ({win_count_w} / {args.n_games})")
    print(f"draw rate = {draw_rate:.2f} %  ({draw_count} / {args.n_games})")
