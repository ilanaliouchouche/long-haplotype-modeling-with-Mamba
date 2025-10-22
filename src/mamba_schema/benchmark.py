import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from src.mamba_schema.mambatrainer import SNPMambaConfig
from src.mamba_models.mambawithpe import MambaLHeadModelWithPETokenClassifier, MambaLHeadModelWithPEPatch, BiMambaLHeadModelWithPEPatch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

class BenchmarkTokenClassification:
    def __init__(self, 
                 model_paths: dict, 
                 dataloaders: dict,
                 test_map: torch.Tensor, 
                 idx2label: dict,
                 positional_encoding: torch.Tensor,
                 model_configs: dict,
                 conv: bool = False,
                 bim: bool = False,
                 pretrained_kwargs: dict = None,
                 log_dir: str = "benchmark",
                 device: str = "cuda:0") -> None:

        self.model_paths = model_paths
        self.dataloaders = dataloaders
        self.test_map = test_map
        self.idx2label = idx2label
        self.positional_encoding = positional_encoding
        self.model_configs = model_configs
        self.pretrained_kwargs = pretrained_kwargs or {}
        self.device = device
        self.conv = conv
        self.bim = bim
        # self.writer = SummaryWriter(log_dir=log_dir)
        self.writers = {model_name: SummaryWriter(log_dir=f"{log_dir}/{model_name}") for model_name in model_paths.keys()}
        self.generations = torch.unique(test_map).tolist()

    def _load_model(self, path: str, model_name: str) -> tuple[nn.Module, int]:
        state_dict = torch.load(path, map_location=self.device)
        config = self.model_configs.get(model_name, SNPMambaConfig())

        if self.bim:
            print("Loading Bi Mamba")
            model = BiMambaLHeadModelWithPEPatch.from_pretrained(
                state_dict=state_dict["model_state_dict"],
                config=config,
                whole_position_encoding=self.positional_encoding,
                device=self.device,
                **self.pretrained_kwargs.get(model_name, {})
            ).to(self.device)
        elif self.conv:
            print("Loading Mamba with Patching")
            model = MambaLHeadModelWithPEPatch.from_pretrained(
                state_dict=state_dict["model_state_dict"],
                config=config,
                whole_position_encoding=self.positional_encoding,
                device=self.device,
                **self.pretrained_kwargs.get(model_name, {})
            ).to(self.device)
        else:
            print("Loading Mamba")
            model = MambaLHeadModelWithPETokenClassifier.from_pretrained(
                state_dict=state_dict["model_state_dict"],
                config=config,
                whole_position_encoding=self.positional_encoding,
                device=self.device,
                **self.pretrained_kwargs.get(model_name, {})
            ).to(self.device)
        return model, state_dict["epoch"]

    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> dict:
        model.eval()
        all_preds, all_labels, all_gens = [], [], []

        with torch.inference_mode():
            for x, y, pos, idxs in tqdm(dataloader, desc="Evaluating", leave=False):
                x, y, pos = x.to(self.device), y.to(self.device), pos.to(self.device)
                logits = model(x, pos)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
                all_gens.append(self.test_map[idxs].cpu())

        return {
            "preds": torch.cat(all_preds, dim=0).numpy(),
            "labels": torch.cat(all_labels, dim=0).numpy(),
            "gens": torch.cat(all_gens, dim=0).numpy()
        }

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray, generations: np.ndarray) -> dict:
        results = {}

        for gen in self.generations:
            mask = generations == gen
            mask = mask.squeeze()
            y_true, y_pred = labels[mask,:].flatten(), preds[mask,:].flatten()
            
            if y_true.size == 0:
                continue

            conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

            results[gen] = {
                "conf_mat": conf_mat,
                "precision": precision, 
                "recall": recall, 
                "f1": f1,
                "macro": (macro_p, macro_r, macro_f1),
                "micro": (micro_p, micro_r, micro_f1),
                "weighted": (weighted_p, weighted_r, weighted_f1),
            }

        return results

    def _log_results(self, model_name: str, epoch: int, metrics: dict) -> None:
        self.writers[model_name].add_scalar(f"epoch", epoch)

        for gen, data in metrics.items():
            self.writers[model_name].add_scalar(f"macro_f1/gen_{gen}", data["macro"][2])
            self.writers[model_name].add_scalar(f"micro_f1/gen_{gen}", data["micro"][2])
            self.writers[model_name].add_scalar(f"weighted_f1/gen_{gen}", data["weighted"][2])
            self._log_confusion_matrix(model_name, gen, data["conf_mat"])

            for i, label in self.idx2label.items():
                self.writers[model_name].add_scalar(f"precision_{label}/gen_{gen}", data["precision"][i])
                self.writers[model_name].add_scalar(f"recall_{label}/gen_{gen}", data["recall"][i])
                self.writers[model_name].add_scalar(f"f1_{label}/gen_{gen}", data["f1"][i])
    
    def _log_confusion_matrix(self, model_name: str, gen: int, conf_mat: np.ndarray):        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", xticklabels=self.idx2label.values(), yticklabels=self.idx2label.values(), ax=ax)
        ax.set_xlabel("Preds")
        ax.set_ylabel("GT")
        ax.set_title(f"Confusion Matrix - {model_name} - Generation {gen}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)


        self.writers[model_name].add_image(f"conf_matrix/gen_{gen}", image_tensor)
        plt.close(fig)

    def run(self) -> None:
        for name, path in self.model_paths.items():
            dataloader = self.dataloaders.get(name)
            if dataloader is None:
                print(f"No DataLoader found for {name}, skipping...")
                continue
            
            model, epoch = self._load_model(path, name)
            eval_results = self._evaluate_model(model, dataloader)
            metrics = self._compute_metrics(eval_results["preds"], eval_results["labels"], eval_results["gens"])
            self._log_results(name, epoch, metrics)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
