# flake8: noqa

from mamba_ssm.models.config_mamba import MambaConfig
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Optional, List, Dict, Literal
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from collections import defaultdict
import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BPE_JSON_PATH = "./bpe_tokens.json"
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "[EOS]"

def AATS(A: np.ndarray, B: np.ndarray, p: int, device: torch.device):
    A_t = torch.tensor(A, dtype=torch.float, device=device)
    B_t = torch.tensor(B, dtype=torch.float, device=device)

    dTT = torch.cdist(A_t, A_t, p=p)
    dTT.fill_diagonal_(torch.inf)
    dTT = dTT.min(dim=1).values

    dAB = torch.cdist(B_t, A_t, p=p)
    dST = dAB.min(dim=1).values
    dTS = dAB.min(dim=0).values

    dSS = torch.cdist(B_t, B_t, p=p)
    dSS.fill_diagonal_(torch.inf)
    dSS = dSS.min(dim=1).values

    n = dSS.shape[0]
    AAtruth = ((dTS > dTT).sum() / n).item()
    AAsyn   = ((dST > dSS).sum() / n).item()
    return AAtruth, AAsyn


@dataclass
class SNPMambaConfig(MambaConfig):
    num_labels : Optional[int] = None


class MambaTrainer:
    def __init__(self,
                 config: SNPMambaConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 test_loader: Optional[DataLoader],
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.modules.loss._Loss,
                 compute_metrics: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]] = lambda x, y, z: {},
                 task: Literal["element_wise", "window_wise", "causal"] = "window_wise",
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 train_step: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
                 val_step: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
                 device: str = "cuda:0",
                 seed: int = 42,
                 mlm: bool = False,
                 n_labels: Optional[int] = None) -> None:
        """
        A training class for PyTorch models, providing support for training, validation, gradient tracking, 
        and TensorBoard logging.

        :param config: Configuration object specific to the training process.
        :type config: SNPMambaConfig
        :param model: The PyTorch model to be trained.
        :type model: torch.nn.Module
        :param train_loader: DataLoader for the training dataset.
        :type train_loader: DataLoader
        :param validation_loader: DataLoader for the validation dataset.
        :type validation_loader: DataLoader
        :param test_loader: Optional DataLoader for the test dataset.
        :type test_loader: Optional[DataLoader]
        :param optimizer: Optimizer for updating the model's parameters.
        :type optimizer: torch.optim.Optimizer
        :param loss_fn: Loss function used during training and validation.
        :type loss_fn: torch.nn.modules.loss._Loss
        :param compute_metrics: A callable for calculating metrics based on predictions and labels.
        :type compute_metrics: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]
        :param task: The task:
            - "element_wise": Classifies each element in the window.
            - "window_wise": Classifies the entire window.
            - "causal": Predicts the next element in the sequence.
        :type task: Literal["element_wise", "window_wise"]
        :param scheduler: Optional learning rate scheduler.
        :type scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
        :param train_step: Optional custom function for the training step. If not provided, a default is used.
        :type train_step: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]]
        :param val_step: Optional custom function for the validation step. If not provided, a default is used.
        :type val_step: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]]
        :param device: Device for computations (e.g., "cuda:0" or "cpu").
        :type device: str
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param n_labels: Optional number of labels (for multi-class classification tasks).
        :type n_labels: Optional[int]
        """

        self.set_seed(seed)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compute_metrics = compute_metrics
        self.task = task
        self.scheduler = scheduler
        self.config = config
        self.n_labels = n_labels
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.mlm = mlm

        if train_step:
            self.train_step = train_step
        else:
            self.train_step = self._make_train_step(mlm=mlm)
        
        if val_step:
            self.val_step = val_step
        else:
            self.val_step = self._make_val_step(mlm=mlm)

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.gradients = defaultdict(lambda: defaultdict(list))
        self.handles = []

        # loooooooooogs
        self.seq_length = 10_000
        self.tokenizer = Tokenizer.from_file(BPE_JSON_PATH)
        self.id_pad_token = self.tokenizer.token_to_id(PAD_TOKEN)
        self.id_eos_token = self.tokenizer.token_to_id(EOS_TOKEN)

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_tensorboard(self, name: str, log_dir: str = "runs") -> None:
        suffix = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{name}_{suffix}")

    def _make_train_step(self, mlm: bool = False) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]:

        if mlm:
            def train_step(x: torch.Tensor, y: torch.Tensor, pos: torch.Tensor, idxs: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                
                yhat = self.model(x, pos, mask)
                try:
                    yhat = yhat.squeeze(1)
                except:
                    yhat = yhat.logits  # case we use CausalLM
                if self.task != "window_wise":
                    yhat = yhat.permute(0, 2, 1)
                loss = self.loss_fn(yhat, y)
                loss.backward()
                self.optimizer.step()
    
                metrics = self.compute_metrics(yhat, y, idxs)
                metrics["loss"] = loss.item()
    
                return metrics
        else:
            def train_step(x: torch.Tensor, y: torch.Tensor, pos: torch.Tensor, idxs: torch.Tensor) -> Dict[str, float]:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                
                yhat = self.model(x, pos)
                try:
                    yhat = yhat.squeeze(1)
                except:
                    yhat = yhat.logits  # case we use CausalLM
                if self.task != "window_wise":
                    yhat = yhat.permute(0, 2, 1)
                loss = self.loss_fn(yhat, y)
                loss.backward()
                self.optimizer.step()
    
                metrics = self.compute_metrics(yhat, y, idxs)
                metrics["loss"] = loss.item()
    
                return metrics

        return train_step

    def _make_val_step(self, mlm: bool = False) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]:
        if mlm:
            def val_step(x: torch.Tensor, y: torch.Tensor, pos: torch.Tensor, idxs: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
                self.model.eval()
                with torch.inference_mode():
                    yhat = self.model(x, pos, mask)
                    try:
                        yhat = yhat.squeeze(1)
                    except:
                        yhat = yhat.logits  # case we use CausalLM
                    if self.task != "window_wise":
                        yhat = yhat.permute(0, 2, 1)
                    loss = self.loss_fn(yhat, y)
    
                    metrics = self.compute_metrics(yhat, y, idxs)
                    metrics["loss"] = loss.item()
    
                    return metrics
    
            return val_step
        else:
            def val_step(x: torch.Tensor, y: torch.Tensor, pos: torch.Tensor, idxs: torch.Tensor) -> Dict[str, float]:
                self.model.eval()
                with torch.inference_mode():
                    yhat = self.model(x, pos)
                    try:
                        yhat = yhat.squeeze(1)
                    except:
                        yhat = yhat.logits  # case we use CausalLM
                    if self.task != "window_wise":
                        yhat = yhat.permute(0, 2, 1)
                    loss = self.loss_fn(yhat, y)
    
                    metrics = self.compute_metrics(yhat, y, idxs)
                    metrics["loss"] = loss.item()
    
                    return metrics
    
            return val_step
            

    def _mini_batch(self, validation: bool = False, mlm: bool = False) -> Dict[str, float]:
        data_loader = self.validation_loader if validation else self.train_loader
        step = self.val_step if validation else self.train_step

        total_metrics = defaultdict(float)
        total_batches = 0
        if mlm:
            for x, y, pos, idxs, mask in tqdm(data_loader, desc="Batch progress", leave=False):
                x, y, pos, mask = x.to(self.device), y.to(self.device), pos.to(self.device), mask.to(self.device)
                metrics = step(x, y, pos, idxs, mask)
                for key, value in metrics.items():
                    total_metrics[key] += value
                total_batches += 1
        else:
            for x, y, pos, idxs in tqdm(data_loader, desc="Batch progress", leave=False):
                x, y, pos = x.to(self.device), y.to(self.device), pos.to(self.device)
                metrics = step(x, y, pos, idxs)
                for key, value in metrics.items():
                    total_metrics[key] += value
                total_batches += 1

        return {key: value / total_batches for key, value in total_metrics.items()}

    def train(self,
              n_epochs: int,
              seed: int = 42,
              checkpoint_step: int = 0,
              checkpoint_path: str = "./checkpoints",
              logging_step: int = 1,
              log_embedding_cos_sim: bool = False) -> None:
        """
        Train the model for n_epochs and optionally save checkpoints and log progress.

        :param n_epochs: Number of epochs to train.
        :type n_epochs: int
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param checkpoint_step: Save model every 'checkpoint_step' epochs (0 to disable).
        :type checkpoint_step: int
        :param checkpoint_path: Path to save model checkpoints.
        :type checkpoint_path: str
        :param logging_step: Print training logs every 'logging_step' epochs.
        :type logging_step: int
        """

        self.set_seed(seed)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"Training for {n_epochs} epochs. {self.total_epochs} epochs have already been trained.")

        for epoch in tqdm(range(n_epochs), desc="Training progress"):
            self.total_epochs += 1

            train_metrics = self._mini_batch(mlm=self.mlm)
            self.losses.append(train_metrics["loss"])

            val_metrics = self._mini_batch(validation=True, mlm=self.mlm)
            self.val_losses.append(val_metrics["loss"])

            if logging_step > 0 and self.total_epochs % logging_step == 0:
                print(f"{self.total_epochs} epochs trained. Train metrics: {train_metrics} | Val metrics: {val_metrics}")

                if hasattr(self, "writer") and self.writer:
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f"Train/{key}", value, epoch)
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"Validation/{key}", value, epoch)
                    
                    if log_embedding_cos_sim:
                        self._log_embedding_cos_sim(self.total_epochs)
                    
                    self._log_gradients(epoch)

                    if self.task == "causal":
                        self._log_aats()
                        self._log_pca()

            if checkpoint_step > 0 and self.total_epochs % checkpoint_step == 0:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                checkpoint_filename = f"checkpoint-{self.total_epochs}-{timestamp}.pth"
                checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_filename)
                self.save_model(checkpoint_full_path)

            if self.scheduler:
                self.scheduler.step()

        print(f"Training completed. {n_epochs} epochs trained. Total epochs trained: {self.total_epochs}")

        if hasattr(self, "writer") and self.writer:
            self.writer.flush()

    def _token_ids_to_snp_array(self, token_seq: np.ndarray) -> np.ndarray:
        eos_positions = np.where(token_seq == self.id_eos_token)[0]
        if len(eos_positions) > 0:
            token_seq = token_seq[: eos_positions[0] ]
        txt = self.tokenizer.decode(token_seq.tolist(), skip_special_tokens=True)
        snps = np.array([int(c) for c in txt], dtype=np.int64)
        L = self.seq_length
        out = np.zeros(L, dtype=np.int64)
        if snps.shape[0] >= L:
            out[:] = snps[:L]
        else:
            out[: snps.shape[0] ] = snps
        return out

    def _prepare_snp_matrices(self):
        real_list = []
        pred_list = []
        self.model.eval()
        with torch.inference_mode():
            for x, y, pos, idxs in self.validation_loader:
                x, pos = x.to(self.device), pos.to(self.device)
                out = self.model(x, pos)
                logits = getattr(out, "logits", out)
                # reshape [B, V, T] -> [B, T, V] si nécessaire
                if logits.dim() == 3 and logits.shape[1] != x.shape[1]:
                    logits = logits.permute(0, 2, 1)

                preds = logits.argmax(dim=-1).cpu().numpy()  # (B, T_batch)
                real  = x.cpu().numpy()                     # (B, T_batch)

                for real_tok_seq, pred_tok_seq in zip(real, preds):
                    real_list.append(self._token_ids_to_snp_array(real_tok_seq))
                    pred_list.append(self._token_ids_to_snp_array(pred_tok_seq))

        real_snps = np.stack(real_list, axis=0)  # (N, seq_length)
        pred_snps = np.stack(pred_list, axis=0)  # (N, seq_length)
        return real_snps, pred_snps

    def _log_aats(self):
        real_snps, pred_snps = self._prepare_snp_matrices()
        aa_truth, aa_syn = AATS(real_snps, pred_snps, p=2, device=self.device)
        self.writer.add_scalar("AATS/AA_truth", aa_truth, self.total_epochs)
        self.writer.add_scalar("AATS/AA_syn",   aa_syn,   self.total_epochs)

    def _log_pca(self):
        real_snps, pred_snps = self._prepare_snp_matrices()
        X = np.vstack([real_snps, pred_snps])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        labels = ["base"] * real_snps.shape[0] + ["synthetic"] * pred_snps.shape[0]
        self.writer.add_embedding(
            torch.tensor(coords, dtype=torch.float),
            metadata=labels,
            global_step=self.total_epochs,
            tag="PCA_2D"
        )


    def _log_gradients(self, epoch: int) -> None:
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                
                param_norm = param.grad.data.norm(2).item()
                self.writer.add_scalar(f"Gradients/{name}_norm", param_norm, epoch)
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5
        self.writer.add_scalar("Gradients/global_norm", total_norm, epoch)

    def hook_gradients(self, layers: List[str]) -> None:
        modules = list(self.model.named_modules())

        def make_log_gd(layer_name: str, param_name: str) -> Callable:
            def log_gd(grad):
                self.gradients[layer_name][param_name].append(grad.detach().cpu().numpy())

            return log_gd

        for name, module in modules:
            if name in layers:
                for param_name, param in module.named_parameters():
                    handle = param.register_hook(make_log_gd(name, param_name))
                    self.handles.append(handle)

    def unhook_gradients(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.gradients = defaultdict(lambda: defaultdict(list))

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.total_epochs,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "config": self.config.__dict__,
            "n_labels": self.n_labels
        }, path)
        print(f"Model saved at {path}")

    def load_model(self, path: str, eval: bool = False) -> None:
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.total_epochs = checkpoint["epoch"]
            self.losses = checkpoint["losses"]
            self.val_losses = checkpoint["val_losses"]
            self.config = SNPMambaConfig(**checkpoint["config"])
            self.n_labels = checkpoint["n_labels"]

            if eval:
                self.model.eval()
            else:
                self.model.train()

            print(f"Model loaded from {path}. Total epochs trained: {self.total_epochs}")
        except FileNotFoundError:
            print(f"Error: File {path} not found.")

    def predict(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            yhat = self.model(x, pos)
            try:
                return torch.argmax(yhat, dim=-1)
            except:
                return torch.argmax(yhat.logits, dim=-1)
    
    def add_graph(self) -> None:
        return
#        try:
#            x, _, pos, _ = next(iter(self.train_loader))
#        except:
#            x, _, pos, _, _ = next(iter(self.train_loader))
#        x, pos = x.to(self.device), pos.to(self.device)
#        try:
#            self.writer.add_graph(self.model, (x, pos))
#        except:
#            print(f"Failed to add graph. Check if model is on the correct device.")
#        finally:
#            del x, pos

    @staticmethod
    def find_embedding_layer(module: nn.Module) -> Optional[nn.Embedding]:
        if isinstance(module, nn.Embedding):
            return module

        for _, child in module.named_children():
            result = MambaTrainer.find_embedding_layer(child)
            if result is not None:
                return result

        return None

    def _log_embedding_cos_sim(self,
                               step: int) -> None:
        embedding_layer = MambaTrainer.find_embedding_layer(self.model)
        if embedding_layer is None:
            return

        embeddings = embedding_layer.weight.data
        if embeddings.shape[0] != 2:  
            return

        emb1 = F.normalize(embeddings[0], dim=0)
        emb2 = F.normalize(embeddings[1], dim=0)

        cos_theta = torch.dot(emb1, emb2).clamp(-1, 1)

        self.writer.add_scalar("Train/embedding_cos_sim", cos_theta.item(), step)
