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


class DataLoader:
    def __init__(self, file_names, batch_size, n_iter, device):
        self.file_names = file_names
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device

        self.entry_size = ctypes.sizeof(Entry)
        self.pos_binary = np.array([1 << i for i in range(64)], dtype=np.uint64)

        self.files = [open(file_name, "rb") for file_name in file_names]
        self.n_entries = []
        for file in self.files:
            file.seek(0, 2)
            self.n_entries.append(file.tell() // self.entry_size)

        self.total_entry = sum(self.n_entries)
        self.iter_count = 0
        # self.max_count = self.total_entry * n_iter // batch_size

        for file_name, n_entry in zip(file_names, self.n_entries):
            print(f"{file_name} : {n_entry}")
        print(f"total : {self.total_entry}")

    def __iter__(self):
        return self
    
    def __len__(self):
        # return self.max_count
        return self.n_iter

    def __next__(self):
        # if self.iter_count == self.max_count:
        if self.iter_count == self.n_iter:
            for file in self.files:
                file.close()
            raise StopIteration()

        black_bitboard_b = []
        white_bitboard_b = []
        side_b = []
        legal_flags_b = []
        result_b = []
        Q_b = []
        posteriors_b = []

        file_idx = np.random.choice(len(self.files))
        file = self.files[file_idx]
        batch_size = min(self.batch_size, self.n_entries[file_idx])
        entry_idxs = np.random.choice(self.n_entries[file_idx], batch_size, replace=False)
        entry_idxs = np.sort(entry_idxs)

        for idx in entry_idxs:
            entry = Entry()
            file.seek(idx * self.entry_size)
            file.readinto(entry)

            black_bitboard_b.append(entry.black_bitboard)
            white_bitboard_b.append(entry.white_bitboard)
            side_b.append(entry.side)
            legal_flags_b.append(entry.legal_flags)
            result_b.append(entry.result)
            Q_b.append(entry.Q)
            posteriors_b.append(entry.posteriors)

        black_bitboard_b = np.array(black_bitboard_b, dtype=np.uint64)
        white_bitboard_b = np.array(white_bitboard_b, dtype=np.uint64)
        black_board_flat_b = (black_bitboard_b[:, None] & self.pos_binary > 0).astype(np.float32)
        white_board_flat_b = (white_bitboard_b[:, None] & self.pos_binary > 0).astype(np.float32)
        black_board_b = black_board_flat_b.reshape((-1, 8, 8))
        white_board_b = white_board_flat_b.reshape((-1, 8, 8))

        black_board_b = torch.tensor(black_board_b, device=self.device)
        white_board_b = torch.tensor(white_board_b, device=self.device)
        side_b = torch.tensor(side_b, dtype=torch.float, device=self.device)
        legal_flags_b = torch.tensor(legal_flags_b, dtype=torch.float, device=self.device)
        result_b = torch.tensor(result_b, dtype=torch.float, device=self.device)
        Q_b = torch.tensor(Q_b, dtype=torch.float, device=self.device)
        posteriors_b = torch.tensor(posteriors_b, dtype=torch.float, device=self.device)
        
        self.iter_count += 1
        return black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b
