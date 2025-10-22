# flake8: noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm

class SNPBeamSampler:
    def __init__(self,
                 model,
                 window_size: int,
                 stride: int,
                 base: int,
                 max_window_index: int,
                 use_idx: bool,
                 beam_size: int = 2,
                 score_mode: str = "global",
                 device=None):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.overlap = window_size - stride
        self.base = base
        self.k_digits = math.ceil(math.log(max_window_index + 1, base))
        self.use_idx = use_idx
        self.beam = beam_size
        self.mode = score_mode
        self.dev = device or next(model.parameters()).device

    def _to_base_tokens(self, j: int):
        digs = []
        x = j
        for _ in range(self.k_digits):
            digs.append(x % self.base)
            x //= self.base
        return torch.tensor(digs[::-1], device=self.dev) + 2

    @torch.inference_mode
    def generate(self,
                 init_seed: torch.LongTensor,
                 total_windows: int,
                 filter_idx: bool = False,
                 pbar:bool = True) -> torch.LongTensor:
        B = init_seed.size(0)
        beams = self.beam
        seed = init_seed.to(self.dev)
        global_scores = torch.zeros(B, beams, device=self.dev)
        seqs = seed.unsqueeze(1).expand(-1, beams, -1).clone()
        best_news = []
        best_idxs = []

        windows = tqdm(range(total_windows), desc="Generating Windows") if pbar else range(total_windows)

        for j in windows:
            if self.mode == 'local':
                scores = torch.zeros(B, beams, device=self.dev)
            else:
                scores = global_scores

            if j > 0:
                seqs = seqs[:, :, -self.overlap:]
            if self.use_idx:
                idx_toks = self._to_base_tokens(j)
                idxs = idx_toks.unsqueeze(0).unsqueeze(1).expand(B, beams, -1)
                seqs = torch.cat([seqs, idxs], dim=2)
                best_idxs.append(idx_toks.unsqueeze(0).expand(B, -1))

            cur_len = seqs.size(2)
            for t in range(self.stride):
                flat = seqs.view(B * beams, cur_len + t)
                logits = self.model(flat).logits[:, -1, :2]
                logp = F.log_softmax(logits, dim=-1)
                logp = logp.view(B, beams, 2)
                scores_exp = scores.unsqueeze(-1) + logp
                scores_flat = scores_exp.view(B, beams * 2)
                top, idx = scores_flat.topk(beams, dim=-1)
                scores = top
                beam_idx = idx // 2
                tok_idx = idx % 2
                seqs = seqs.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, cur_len + t))
                seqs = torch.cat([seqs, tok_idx.unsqueeze(-1)], dim=2)

            best_news.append(seqs[:, 0, -self.stride:].clone())
            if self.mode == 'global':
                global_scores = scores

        if filter_idx:
            return torch.cat(best_news, dim=1)

        parts = [seed]
        for i in range(total_windows):
            if self.use_idx:
                parts.append(best_idxs[i])
            parts.append(best_news[i])
        return torch.cat(parts, dim=1)
