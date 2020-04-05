import numpy as np
import ctypes
import torch
import pathlib
import re
import time


class Entry(ctypes.Structure):
    _fields_ = [
        ("black_bitboard", ctypes.c_uint64),
        ("white_bitboard", ctypes.c_uint64),
        ("side", ctypes.c_uint8),
        ("action", ctypes.c_uint8),
        ("Q", ctypes.c_float),
        ("result", ctypes.c_float),
        ("legal_flags", ctypes.c_bool * 64),
        ("posteriors", ctypes.c_float * 64),
    ]


class DataLoader():
    def __init__(self, mldata_dir_path, window_size, batch_size, n_update, n_thread):
        self.mldata_dir_path = mldata_dir_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_update = n_update
        self.n_thread = n_thread

        self.mldata_pool = {}
        self.select_ratios = {}

        self.game_count = 0
        # if there are merged mldata files, game_count is set to the oldest of them
        merged_counts = []
        for path in mldata_dir_path.glob(f"*.dat"):
            m = re.search(r"/([0-9]+).dat", path.name)
            if m:
                merged_counts.append(int(m.group(1)))
        if len(merged_counts) > 0:
            self.game_count = min(merged_counts)
        print(f"PY/MLDATA  game_count intitlized to {self.game_count}")

        self.update_count = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.update_count >= self.n_update:
            self.update_count = 0
            raise StopIteration()

        self.update_count += 1

        # select pool key (newer selected more frequently)
        pool_keys = list(self.mldata_pool.keys())
        select_probs = np.array([self.select_ratios[key] for key in pool_keys])
        select_probs = select_probs / select_probs.sum()
        pool_key = np.random.choice(pool_keys, p=select_probs)

        data = self.mldata_pool[pool_key]
        entry_idxs = np.random.choice(len(data["black_board"]), self.batch_size, replace=False)

        black_board_b = data["black_board"][entry_idxs]
        white_board_b = data["white_board"][entry_idxs]
        side_b = data["side"][entry_idxs]
        legal_flags_b = data["legal_flags"][entry_idxs]
        result_b = data["result"][entry_idxs]
        Q_b = data["Q"][entry_idxs]
        posteriors_b = data["posteriors"][entry_idxs]

        return black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b


    def load_data(self):
        print(f"PY/MLDATA  load data start")

        while True:
            new_path = self.mldata_dir_path / f"{self.game_count}.dat"
            if not new_path.exists():
                new_sub_paths = list(self.mldata_dir_path.glob(f"{self.game_count}_*.dat"))

                # new data not arrived yet
                if len(new_sub_paths) < self.n_thread:
                    if len(self.mldata_pool) == 0:  # no data available
                        print("PY/MLDATA  waiting for data...")
                        time.sleep(2)
                        continue
                    else:  # old data available
                        print("PY/MLDATA  load data finish")
                        break

                # new data arrived (not merged yet)
                print("PY/MLDATA  merge mldata files")
                with open(new_path, "wb") as output_file:
                    for input_path in new_sub_paths:
                        with open(input_path, "rb") as input_file:
                            output_file.write(input_file.read())
                        input_path.unlink()

            print(f"PY/MLDATA  add mldata {new_path.name}")
            self.mldata_pool[new_path.name] = self.load_file(new_path)
            # print(f"PY/MLDATA  len = {len(self.mldata_pool[new_path.name]['black_board'])}")

            if self.game_count >= self.window_size:
                delete_path = self.mldata_dir_path / f"{self.game_count - self.window_size}.dat"
                if delete_path.exists():
                    print(f"PY/MLDATA  unlink mldata {delete_path.name}")
                    delete_path.unlink()
                    # delete dict element if exists
                    self.mldata_pool.pop(delete_path.name, None)
                    self.select_ratios.pop(delete_path.name, None)

            self.game_count += 1

            # TODO: prob ok?
            for key in self.select_ratios.keys():
                self.select_ratios[key] *= 0.99
            self.select_ratios[new_path.name] = 1.0
            print(self.select_ratios)


    def load_file(self, file_path):
        print(f"PY/MLDATA  load file {file_path.name}")
        pos_binary = np.array([1 << i for i in range(64)], dtype=np.uint64)

        black_bitboard = []
        white_bitboard = []
        side = []
        legal_flags = []
        result = []
        Q = []
        posteriors = []

        with open(file_path, "rb") as file:
            entry = Entry()
            # print(f"PY/MLDATA  load data from {file_path}")
            while file.readinto(entry):
                black_bitboard.append(entry.black_bitboard)
                white_bitboard.append(entry.white_bitboard)
                side.append(entry.side)
                legal_flags.append(np.ctypeslib.as_array(entry.legal_flags).copy())
                result.append(entry.result)
                Q.append(entry.Q)
                # TODO: how to set policy target? (tau=0)
                # posteriors_st = np.ctypeslib.as_array(entry.posteriors)
                # posteriors = np.zeros_like(posteriors_st)
                # posteriors[posteriors_st.argmax()] = 1.0
                posteriors.append(np.ctypeslib.as_array(entry.posteriors).copy())

            # n_entries.append(file.tell() // ctypes.sizeof(Entry))

        black_bitboard = np.array(black_bitboard).astype(np.uint64)
        white_bitboard = np.array(white_bitboard).astype(np.uint64)
        black_board_flat = (black_bitboard[:, None] & pos_binary > 0).astype(np.float32)
        white_board_flat = (white_bitboard[:, None] & pos_binary > 0).astype(np.float32)
        black_board = black_board_flat.reshape((-1, 8, 8))
        white_board = white_board_flat.reshape((-1, 8, 8))

        data = {
            "black_board": torch.tensor(black_board),
            "white_board": torch.tensor(white_board),
            "side": torch.tensor(side, dtype=torch.float),
            "legal_flags": torch.tensor(legal_flags, dtype=torch.float),
            "result": torch.tensor(result, dtype=torch.float),
            "Q": torch.tensor(Q, dtype=torch.float),
            "posteriors": torch.tensor(posteriors, dtype=torch.float),
        }
        return data
