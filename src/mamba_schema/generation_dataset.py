import torch
from torch.utils.data import Dataset
import math
from typing import Callable, Optional, Tuple, Literal, Dict, List
from tqdm import tqdm

class SlidingGenerationDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 sliding_strategy: Callable[[Optional[int]], Tuple[int, int]],
                 embedding_dim: int,
                 base: int,
                 max_window_index: int,
                 position_strategy: Literal["absolute", "first_element", "relative_window"] = "absolute") -> None:

        assert position_strategy in ["absolute", "first_element", "relative_window"], \
            "position_strategy must be 'absolute', 'first_element' or 'relative_window'"

        self._data = data
        self._sliding_strategy = sliding_strategy
        self._embedding_dim = embedding_dim
        self._position_strategy = position_strategy
        self._base = base
        self._max_window_index = max_window_index

        self._vocab_offset = 2  # 0 and 1 for SNPs others for IDX digits
        self._k_digits = int(math.ceil(math.log(max_window_index + 1) / math.log(base)))
        self._id_to_token = self._build_vocab()

        self._position_encodings = self.positional_encoding(data.size(1), embedding_dim)

        self._augmented = False
        self._data, self._labels, self._position_encodings_indices, self._sequence_indices = self._augment_data()

    def _build_vocab(self) -> Dict[int, str]:
        id_to_token = {0: '0', 1: '1'}
        for i in range(self._base):
            id_to_token[self._vocab_offset + i] = f"[IDX_{i}]"
        return id_to_token

    @staticmethod
    def positional_encoding(seq_len: int, embedding_dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _to_base_digits(self, number: int) -> List[int]:
        digits = []
        while number:
            digits.append(number % self._base)
            number //= self._base
        digits = digits[::-1]  # big-endian (most significant first)
        # pad with 0s to ensure fixed length k
        while len(digits) < self._k_digits:
            digits.insert(0, 0)
        return digits

    def _generate_position_indices(self, indices: torch.Tensor, window_idx: int, window_size: int) -> torch.Tensor:
        if self._position_strategy == "absolute":
            return indices
        elif self._position_strategy == "first_element":
            return indices[0].repeat(window_size)
        elif self._position_strategy == "relative_window":
            return torch.full((window_size,), window_idx, dtype=torch.long)
        else:
            raise ValueError(f"Invalid position_strategy: {self._position_strategy}")

    def _augment_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._augmented:
            return self._data, self._labels, self._position_encodings_indices, self._sequence_indices

        all_windows = []
        all_labels = []
        all_positions = []
        all_sequence_indices = []

        for i in tqdm(range(len(self._data)), desc="Generating sliding windows"):
        # for i in range(len(self._data)):
            sample = self._data[i]
            window_size, step = self._sliding_strategy(sample.size(0))
            windows = sample.unfold(dimension=0, size=window_size, step=step)
            n_windows = windows.size(0)

            positions = torch.arange(0, sample.size(0))
            indices = positions.unfold(dimension=0, size=window_size, step=step)

            # for j in tqdm(range(n_windows), desc="Processing windows", leave=False):
            for j in range(n_windows):
                window = windows[j]
                label_window = window.clone()

                index_digits = self._to_base_digits(j)
                digit_tokens = torch.tensor([d + self._vocab_offset for d in index_digits], dtype=torch.long)

                input_ids = torch.cat([digit_tokens, window], dim=0)
                full_sequence = torch.cat([digit_tokens, window], dim=0)
                label_ids = torch.cat([full_sequence[1:], torch.tensor([-100], dtype=torch.long)])

                pos_indices = self._generate_position_indices(indices[j], j, window_size)

                all_windows.append(input_ids)
                all_labels.append(label_ids)
                all_positions.append(pos_indices)
                all_sequence_indices.append(i)

        augmented_data = torch.stack(all_windows)
        augmented_labels = torch.stack(all_labels)
        augmented_positions = torch.stack(all_positions)
        augmented_sequence_indices = torch.tensor(all_sequence_indices)

        self._augmented = True
        return augmented_data, augmented_labels, augmented_positions, augmented_sequence_indices

    def decode_sequence(self, input_ids: torch.Tensor) -> List[str]:
        return [self._id_to_token[int(tok)] for tok in input_ids]

    @property
    def position_encodings(self) -> torch.Tensor:
        return self._position_encodings

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx], self._position_encodings_indices[idx], self._sequence_indices[idx]



def compute_num_windows(seq_len: int, window_size: int, step_size: int) -> int:
    if seq_len < window_size:
        return 0
    return 1 + (seq_len - window_size) // step_size
