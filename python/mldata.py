import numpy as np
import ctypes
import torch


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
    def __init__(self, file_names, batch_size):
        self.file_names = file_names
        self.batch_size = batch_size
        self.entry_size = ctypes.sizeof(Entry)
        self.pos_binary = np.array([1 << i for i in range(64)], dtype=np.uint64)

        n_entries = []
        black_bitboard_all = []
        white_bitboard_all = []
        side_all = []
        legal_flags_all = []
        result_all = []
        Q_all = []
        posteriors_all = []

        for file_name in file_names:
            with open(file_name, "rb") as file:
                entry = Entry()
                # print(f"load data from {file_name}")
                while file.readinto(entry):
                    black_bitboard_all.append(entry.black_bitboard)
                    white_bitboard_all.append(entry.white_bitboard)
                    side_all.append(entry.side)
                    legal_flags_all.append(np.ctypeslib.as_array(entry.legal_flags).copy())
                    result_all.append(entry.result)
                    Q_all.append(entry.Q)
                    # TODO: how to set policy target? (tau=0)
                    # posteriors = np.ctypeslib.as_array(entry.posteriors)
                    # posteriors_st = np.ctypeslib.as_array(entry.posteriors)
                    # posteriors = np.zeros_like(posteriors_st)
                    # posteriors[posteriors_st.argmax()] = 1.0
                    posteriors_all.append(np.ctypeslib.as_array(entry.posteriors).copy())

                n_entries.append(file.tell() // self.entry_size)

        black_bitboard_all = np.array(black_bitboard_all).astype(np.uint64)
        white_bitboard_all = np.array(white_bitboard_all).astype(np.uint64)
        black_board_flat_all = (black_bitboard_all[:, None] & self.pos_binary > 0).astype(np.float32)
        white_board_flat_all = (white_bitboard_all[:, None] & self.pos_binary > 0).astype(np.float32)
        black_board_all = black_board_flat_all.reshape((-1, 8, 8))
        white_board_all = white_board_flat_all.reshape((-1, 8, 8))

        self.black_board_all = torch.tensor(black_board_all)
        self.white_board_all = torch.tensor(white_board_all)
        self.side_all = torch.tensor(side_all, dtype=torch.float)
        self.legal_flags_all = torch.tensor(legal_flags_all, dtype=torch.float)
        self.result_all = torch.tensor(result_all, dtype=torch.float)
        self.Q_all = torch.tensor(Q_all, dtype=torch.float)
        self.posteriors_all = torch.tensor(posteriors_all, dtype=torch.float)

        self.total_entry = sum(n_entries)
        for file_name, n_entry in zip(file_names, n_entries):
            print(f"{file_name} : {n_entry}")
        print(f"total : {self.total_entry}")

        self.position = 0
        self.perm = torch.randperm(self.total_entry)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= self.total_entry:
            self.position = 0
            self.perm = torch.randperm(self.total_entry)
            raise StopIteration()

        idxs = self.perm[self.position:self.position+self.batch_size]
        self.position += self.batch_size

        black_board_b = self.black_board_all[idxs]
        white_board_b = self.white_board_all[idxs]
        side_b = self.side_all[idxs]
        legal_flags_b = self.legal_flags_all[idxs]
        result_b = self.result_all[idxs]
        Q_b = self.Q_all[idxs]
        posteriors_b = self.posteriors_all[idxs]

        return black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b
