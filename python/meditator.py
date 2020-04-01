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
import time
import re
import glob
import os
import datetime


def flip_side(side):
    if side == "b":
        return "w"
    else:
        return "b"


def read_until(proc, phrase):
    contents = ""
    while True:
        line = proc.stdout.readline()
        # print(f"read_until : \"{line}\"")
        if len(line) == 0:
            # print("read_until: disconnected")
            break

        contents += line

        if phrase in contents:
            break

    return contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd1", type=str, help="command for the first program")
    parser.add_argument("cmd2", type=str, help="command for the second program")
    parser.add_argument("-n", "--num-games", type=int, default=1)
    args = parser.parse_args()

    print("side ([b]lack / [w]hite) of the first program ? ", end="")
    side = input()
    if side == "b":
        side1, side2 = "b", "w"
    elif side == "w":
        side1, side2 = "w", "b"
    else:
        raise RuntimeError(f"side ({side}) is invalid")

    print(f"side1={side1} side2={side2}")

    proc1 = subprocess.Popen(args.cmd1.strip().split(" "), encoding='UTF-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(args.cmd2.strip().split(" "), encoding='UTF-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # new line is necessary for any prompt (otherwise buffered)
    side_prompt = "@ [b]lack / [w]hite ?"
    action_prompt = "@ action ?"
    action_output = "@ action :"
    result_output = "@ result :"

    # read until side prompt
    contents = read_until(proc1, side_prompt)
    print(f"proc1 contents = \"{contents}\"")
    contents = read_until(proc2, side_prompt)
    print(f"proc2 contents = \"{contents}\"")

    # send *player* side (not computer side!)
    proc1.stdin.write(side2 + "\n")
    proc1.stdin.flush()
    proc2.stdin.write(side1 + "\n")
    proc2.stdin.flush()

    # start game
    side = "b"
    history = []
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(f"log_{time_stamp}.txt", "w")

    while True:
        if side == side1:  # com of proc1 is in turn
            # print("proc1 in turn")
            proc_send, proc_recv = proc1, proc2
        else:
            # print("proc2 in turn")
            proc_send, proc_recv = proc2, proc1

        # receive action
        contents = read_until(proc_send, action_output)
        # print(f"received: \"{contents}\"")
        m_action = re.search(rf"{action_output}\s*(\w+)", contents)

        if m_action:
            action = m_action.group(1)
            print(f"side={side}, action=\"{action}\"")
            history.append(action)
            log_file.write(action + "\n")
        else:
            # action not received => 2 cases
            # 1) result is received (from both programs)  2) error

            m_result = re.search(rf"{result_output}\s*black\s*=\s*([0-9]+)\s+white\s*=\s*([0-9]+)", contents)
            if m_result:
                result_send = (int(m_result.group(1)), int(m_result.group(2)))
            else:
                if side == side1:
                    print("disconnected unexpectedly by the first program")
                else:
                    print("disconnected unexpectedly by the second program")
                exit(-1)

            contents = read_until(proc_recv, result_output)
            m_result = re.search(rf"{result_output}\s*black\s*=\s*([0-9]+)\s+white\s*=\s*([0-9]+)", contents)
            if m_result:
                result_recv = (int(m_result.group(1)), int(m_result.group(2)))
            else:
                if side == side1:
                    print("disconnected unexpectedly by the first program")
                else:
                    print("disconnected unexpectedly by the second program")
                exit(-1)

            if side == side1:
                result1, result2 = result_send, result_recv
            else:
                result1, result2 = result_recv, result_send

            print(f"result from the first program: black={result1[0]} white={result1[1]}")
            print(f"result from the first program: black={result2[0]} white={result2[1]}")
            break

        # send action
        contents = read_until(proc_recv, action_prompt)
        # print(f"not in turn contents = \"{contents}\"")
        proc_recv.stdin.write(action + "\n")
        proc_recv.stdin.flush()
        print(f"send action \"{action}\"")

        side = flip_side(side)
    
    proc1.wait()
    proc2.wait()

    # print(history)
    log_file.close()
    print("finish!")
