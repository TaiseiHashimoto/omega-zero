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


def make_all_variations(board):
    return torch.stack([
        board,
        board.flip(1),
        board.flip(2),
        board.permute(0, 2, 1),
        board.flip(1).permute(0, 2, 1).flip(1),
        torch.rot90(board, 1, (1, 2)),
        torch.rot90(board, 2, (1, 2)),
        torch.rot90(board, 3, (1, 2)),
    ], dim=0)


def make_variation(board, norm_idx):
    board_vars = make_all_variations(board)
    return board_vars[norm_idx, range(len(board))]


def normalize_input(black_board, white_board):
    position_board = torch.linspace(0, 1, 64).view(8, 8)
    black_board_vars = make_all_variations(black_board)
    white_board_vars = make_all_variations(white_board)

    score = ((black_board_vars + white_board_vars) * position_board).sum(dim=(-1, -2))
    norm_idx = torch.argmax(score, dim=0)

    batch_range = torch.arange(len(black_board))
    return black_board_vars[norm_idx, batch_range], white_board_vars[norm_idx, batch_range], norm_idx



class DataLoader():
    def __init__(self, file_paths, batch_size):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.entry_size = ctypes.sizeof(Entry)
        self.pos_binary = np.array([1 << i for i in range(64)], dtype=np.uint64)

        black_board_all = []
        white_board_all = []
        side_all = []
        legal_flags_all = []
        result_all = []
        Q_all = []
        posteriors_all = []

        for file_path in file_paths:
            bu_path = file_path.with_suffix(".pt")

            if bu_path.exists():  # use backup
                print(f"load {file_path.name} from backup", end="  ")
                data = torch.load(bu_path)
                black_board_file = data["black_board"]
                white_board_file = data["white_board"]
                side_file = data["side"]
                legal_flags_file = data["legal_flags"]
                result_file = data["result"]
                Q_file = data["Q"]
                posteriors_file = data["posteriors"]
            else:
                print(f"load {file_path.name} from data file", end="  ")
                black_bitboard_file = []
                white_bitboard_file = []
                side_file = []
                legal_flags_file = []
                result_file = []
                Q_file = []
                posteriors_file = []
                with open(file_path, "rb") as file:
                    entry = Entry()
                    # print(f"load data from {file_path}")
                    while file.readinto(entry):
                        black_bitboard_file.append(entry.black_bitboard)
                        white_bitboard_file.append(entry.white_bitboard)
                        side_file.append(entry.side)
                        legal_flags_file.append(np.ctypeslib.as_array(entry.legal_flags).copy())
                        result_file.append(entry.result)
                        Q_file.append(entry.Q)
                        # TODO: how to set policy target? (tau=0)
                        # posteriors = np.ctypeslib.as_array(entry.posteriors)
                        # posteriors_st = np.ctypeslib.as_array(entry.posteriors)
                        # posteriors = np.zeros_like(posteriors_st)
                        # posteriors[posteriors_st.argmax()] = 1.0
                        posteriors_file.append(np.ctypeslib.as_array(entry.posteriors).copy())

                black_bitboard_file = np.array(black_bitboard_file).astype(np.uint64)
                white_bitboard_file = np.array(white_bitboard_file).astype(np.uint64)
                black_board_flat_file = (black_bitboard_file[:, None] & self.pos_binary > 0).astype(np.float32)
                white_board_flat_file = (white_bitboard_file[:, None] & self.pos_binary > 0).astype(np.float32)
                black_board_file = black_board_flat_file.reshape((-1, 8, 8))
                white_board_file = white_board_flat_file.reshape((-1, 8, 8))

                black_board_file = torch.tensor(black_board_file)
                white_board_file = torch.tensor(white_board_file)
                side_file = torch.tensor(side_file, dtype=torch.float)
                legal_flags_file = torch.tensor(legal_flags_file, dtype=torch.float)
                result_file = torch.tensor(result_file, dtype=torch.float)
                Q_file = torch.tensor(Q_file, dtype=torch.float)
                posteriors_file = torch.tensor(posteriors_file, dtype=torch.float)

                # normalize board direction
                black_board_file, white_board_file, norm_idx = normalize_input(black_board_file, white_board_file)
                legal_flags_file = make_variation(legal_flags_file.view(-1, 8, 8), norm_idx).view(-1, 64)
                posteriors_file = make_variation(posteriors_file.view(-1, 8, 8), norm_idx).view(-1, 64)

                torch.save({
                    "black_board": black_board_file,
                    "white_board": white_board_file,
                    "side": side_file,
                    "legal_flags": legal_flags_file,
                    "result": result_file,
                    "Q": Q_file,
                    "posteriors": posteriors_file,
                }, bu_path)

            print(f"size: {len(black_board_file)}")

            black_board_all.append(black_board_file)
            white_board_all.append(white_board_file)
            side_all.append(side_file)
            legal_flags_all.append(legal_flags_file)
            result_all.append(result_file)
            Q_all.append(Q_file)
            posteriors_all.append(posteriors_file)

        self.black_board_all = torch.cat(black_board_all, dim=0)
        self.white_board_all = torch.cat(white_board_all, dim=0)
        self.side_all = torch.cat(side_all, dim=0)
        self.legal_flags_all = torch.cat(legal_flags_all, dim=0)
        self.result_all = torch.cat(result_all, dim=0)
        self.Q_all = torch.cat(Q_all, dim=0)
        self.posteriors_all = torch.cat(posteriors_all, dim=0)

        self.total_entry = len(self.black_board_all)
        print(f"total size : {self.total_entry}")

        self.position = 0
        self.perm = torch.randperm(self.total_entry)


    def __iter__(self):
        return self

    def __len__(self):
        return (self.total_entry + self.batch_size - 1) // self.batch_size

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
