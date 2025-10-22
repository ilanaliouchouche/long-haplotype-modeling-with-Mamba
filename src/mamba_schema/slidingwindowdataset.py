# flake8: noqa

import math
from typing import Callable, Optional, Tuple, Literal
import torch    
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

class LightSlidingWindowDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 labels: torch.Tensor,
                 sliding_strategy: Callable[[Optional[int]], Tuple[int, int]],
                 embedding_dim: int,
                 classification_mode: Literal["element_wise", "window_wise"] = "window_wise",
                 position_strategy: Literal["absolute", "first_element", "relative_window"] = "absolute"
                 ) -> None:
        
        assert position_strategy in ["absolute", "first_element", "relative_window"], \
            "position_strategy must be 'absolute', 'first_element' or 'relative_window'"

        self._data = data
        self._labels = labels
        self._sliding_strategy = sliding_strategy
        self._embedding_dim = embedding_dim
        self._classification_mode = classification_mode
        self._position_strategy = position_strategy

        self._position_encodings = self.positional_encoding(data.size(1), embedding_dim)

        self._augmented = False
        self._data, self._labels, self._position_encodings_indices, self._sequence_indices = self._augment_data()

    @staticmethod
    def positional_encoding(seq_len: int, embedding_dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, embedding_dim)  # (seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))  # (embedding_dim/2)

        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, embedding_dim/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (seq_len, embedding_dim/2)

        return pe

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
            sample = self._data[i]  # (sequence_length,)
            label_sequence = self._labels[i]  # (sequence_length,)

            window_size, step = self._sliding_strategy(sample.size(0))
            windows = sample.unfold(dimension=0, size=window_size, step=step)  # (n_windows, window_size)
            label_windows = label_sequence.unfold(dimension=0, size=window_size, step=step)  # (n_windows, window_size)

            positions = torch.arange(0, sample.size(0))  # (sequence_length)
            indices = positions.unfold(dimension=0, size=window_size, step=step)  # (n_windows, window_size)

            n_windows = windows.size(0)

            for j in range(n_windows):
                window = windows[j]
                label_window = label_windows[j]

                if self._classification_mode == "window_wise":
                    majority_label = torch.mode(label_window).values
                    all_labels.append(majority_label)
                elif self._classification_mode == "element_wise":
                    all_labels.append(label_window.tolist())

                pos_indices = self._generate_position_indices(indices[j], j, window_size)

                all_windows.append(window)
                all_positions.append(pos_indices)
                all_sequence_indices.append(i)

        augmented_data = torch.stack(all_windows)  # (n_samples, window_size)
        augmented_labels = torch.tensor(all_labels)  # (n_samples,) or (n_samples, window_size)
        augmented_positions = torch.stack(all_positions)  # (n_samples, window_size)
        augmented_sequence_indices = torch.tensor(all_sequence_indices)  # (n_samples,)

        self._augmented = True
        return augmented_data, augmented_labels, augmented_positions, augmented_sequence_indices
    
    @property
    def position_encodings(self) -> torch.Tensor:
        return self._position_encodings

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx], self._position_encodings_indices[idx], self._sequence_indices[idx]
    
    
    @property
    def position_encodings(self) -> torch.Tensor:
        return self._position_encodings

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx], self._position_encodings_indices[idx], self._sequence_indices[idx]


class SlidingWindowCausalDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 sliding_strategy: Callable[[Optional[int]], Tuple[int, int]],
                 embedding_dim: int,
                 position_strategy: Literal["absolute", "first_element", "relative_window"] = "absolute"
                 ) -> None:
        assert position_strategy in ["absolute", "first_element", "relative_window"], \
            "position_strategy must be 'absolute', 'first_element' or 'relative_window'"

        self._data = data
        self._sliding_strategy = sliding_strategy
        self._embedding_dim = embedding_dim
        self._position_strategy = position_strategy

        self._position_encodings = self.positional_encoding(data.size(1), embedding_dim)

        self._augmented = False
        self._data, self._labels, self._position_encodings_indices, self._sequence_indices = self._augment_data()

    @staticmethod
    def positional_encoding(seq_len: int, embedding_dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, embedding_dim)  
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))  

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  

        return pe

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
            sample = self._data[i]  # (sequence_length,)

            window_size, step = self._sliding_strategy(sample.size(0))
            windows = sample.unfold(dimension=0, size=window_size, step=step)  # (n_windows, window_size)
            n_windows = windows.size(0)

            positions = torch.arange(0, sample.size(0))  
            indices = positions.unfold(dimension=0, size=window_size, step=step)  

            for j in tqdm(range(n_windows), desc="Processing windows", leave=False):
                window = windows[j]
                
                label_window = torch.cat([window[1:], torch.tensor([-100], dtype=torch.long)], dim=0)

                pos_indices = self._generate_position_indices(indices[j], j, window_size)

                all_windows.append(window)
                all_labels.append(label_window)
                all_positions.append(pos_indices)
                all_sequence_indices.append(i)

        augmented_data = torch.stack(all_windows)  
        augmented_labels = torch.stack(all_labels)  
        augmented_positions = torch.stack(all_positions)  
        augmented_sequence_indices = torch.tensor(all_sequence_indices)  

        self._augmented = True
        return augmented_data, augmented_labels, augmented_positions, augmented_sequence_indices
    
    @property
    def position_encodings(self) -> torch.Tensor:
        return self._position_encodings

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx], self._position_encodings_indices[idx], self._sequence_indices[idx]
    

def get_position_encoding(sequence_pe: torch.Tensor,
                          indices: torch.Tensor,
                          shuffle: bool = False,
                          total_seq_len: Optional[int] = 10_000,
                          seed: int = 42) -> torch.Tensor:
    if shuffle:
        if total_seq_len is None:
            raise ValueError("total_seq_len must be provided when shuffle is True.")
        
        torch.manual_seed(seed)
        indices = torch.randint(0, total_seq_len, indices.shape, device=indices.device)
    
    # indices shape: (n_samples, window_size)
    sequence_pe = sequence_pe.to(indices.device)
    return sequence_pe[indices]  # (n_samples, window_size, embedding_dim)

def compute_daf_matrix(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    labels = torch.unique(y)
    num_classes = labels.numel()
    Y_onehot = F.one_hot(y, num_classes=num_classes).permute(2, 0, 1)
    X_exp = X.unsqueeze(0)
    num = (Y_onehot * X_exp).sum(dim=1)
    denom = Y_onehot.sum(dim=1).float() + 1e-8
    freqs = num.float() / denom

    return freqs

class DAFSlidingWindowDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        daf_matrix: torch.Tensor,
        sliding_strategy: Callable[[Optional[int]], Tuple[int, int]],
        embedding_dim: int,
        classification_mode: Literal["element_wise", "window_wise"] = "window_wise",
        position_strategy: Literal["absolute", "first_element", "relative_window"] = "absolute",
        concat_snp: bool = False
    ) -> None:

        assert position_strategy in ["absolute", "first_element", "relative_window"]

        self._data = data
        self._labels = labels
        self._sliding_strategy = sliding_strategy
        self._embedding_dim = embedding_dim
        self._classification_mode = classification_mode
        self._position_strategy = position_strategy
        self._concat_snp = concat_snp
        self._daf_matrix = daf_matrix

        self._position_encodings = self.positional_encoding(data.size(1), embedding_dim)

        self._augmented = False
        self._data, self._labels, self._position_encodings_indices, self._sequence_indices = self._augment_data()

    @staticmethod
    def positional_encoding(seq_len: int, embedding_dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _generate_position_indices(self, indices: torch.Tensor, window_idx: int, window_size: int) -> torch.Tensor:
        if self._position_strategy == "absolute":
            return indices
        elif self._position_strategy == "first_element":
            return indices[0].repeat(window_size)
        elif self._position_strategy == "relative_window":
            return torch.full((window_size,), window_idx, dtype=torch.long)
        else:
            raise ValueError(f"Invalid position_strategy: {self._position_strategy}")

    def _augment_data(self):
        if self._augmented:
            return self._data, self._labels, self._position_encodings_indices, self._sequence_indices

        all_windows = []
        all_labels = []
        all_positions = []
        all_sequence_indices = []

        for i in tqdm(range(len(self._data)), desc="Generating sliding windows"):
            sample = self._data[i]
            label_sequence = self._labels[i]

            window_size, step = self._sliding_strategy(sample.size(0))
            windows = sample.unfold(dimension=0, size=window_size, step=step)
            label_windows = label_sequence.unfold(dimension=0, size=window_size, step=step)
            positions = torch.arange(0, sample.size(0))
            indices = positions.unfold(dimension=0, size=window_size, step=step)

            n_windows = windows.size(0)

            for j in range(n_windows):
                label_window = label_windows[j]
                pos_indices = self._generate_position_indices(indices[j], j, window_size)
                seq_idx = i

                indices_j = indices[j]
                x_vals = sample[indices_j]
                daf_window = torch.where(x_vals == 1, self._daf_matrix[:, indices_j], 1 - self._daf_matrix[:, indices_j])

                if self._concat_snp:
                    x_window = torch.cat([x_vals.unsqueeze(0), daf_window], dim=0)
                else:
                    x_window = daf_window

                if self._classification_mode == "window_wise":
                    majority_label = torch.mode(label_window).values
                    all_labels.append(majority_label)
                elif self._classification_mode == "element_wise":
                    all_labels.append(label_window.tolist())

                all_windows.append(x_window)
                all_positions.append(pos_indices)
                all_sequence_indices.append(seq_idx)

        augmented_data = torch.stack(all_windows)
        augmented_labels = torch.tensor(all_labels)
        augmented_positions = torch.stack(all_positions)
        augmented_sequence_indices = torch.tensor(all_sequence_indices)

        self._augmented = True
        return augmented_data, augmented_labels, augmented_positions, augmented_sequence_indices

    @property
    def position_encodings(self) -> torch.Tensor:
        return self._position_encodings

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx], self._labels[idx], self._position_encodings_indices[idx], self._sequence_indices[idx]


def mlm_collate_fn_masked_embedding(batch, mask_prob=0.15):
    input_ids, labels, pos_indices, seq_ids = zip(*batch)

    input_ids = torch.stack(input_ids, dim=0)       # (batch_size, channels, seq_len)
    labels = torch.stack(labels, dim=0)             # (batch_size, seq_len)
    pos_indices = torch.stack(pos_indices, dim=0)   # (batch_size, seq_len)
    seq_ids = torch.stack(seq_ids, dim=0)           # (batch_size,)

    mask = torch.rand(labels.shape, device=labels.device) < mask_prob

    final_labels = labels.clone()
    final_labels[~mask] = -100

    return input_ids, final_labels, pos_indices, seq_ids, mask
