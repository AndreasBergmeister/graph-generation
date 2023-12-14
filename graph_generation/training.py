import pickle
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as th
from hydra.core.hydra_config import HydraConfig
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torch.optim import Adam

import wandb

from .metrics import Metric
from .model import EMA, EMA1


class Trainer:
    def __init__(
        self,
        sign_net,
        model,
        method,
        train_dataloader,
        train_graphs: list[nx.Graph],
        validation_graphs: list[nx.Graph],
        test_graphs: list[nx.Graph],
        metrics: list[Metric],
        cfg,
    ):
        self.train_iterator = iter(train_dataloader)
        self.train_graphs = train_graphs
        self.validation_graphs = validation_graphs
        self.test_graphs = test_graphs
        self.metrics = metrics
        self.cfg = cfg

        self.rng = np.random.default_rng(0)
        self.device = "cuda" if th.cuda.is_available() and not cfg.debugging else "cpu"
        self.method = method.to(self.device)
        self.sign_net = sign_net.to(self.device) if sign_net is not None else None
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), cfg.training.lr)

        # EMA
        self.ema_models = {
            beta: EMA(
                model=self.model, beta=beta, gamma=cfg.ema.gamma, power=cfg.ema.power
            )
            if beta != 1
            else EMA1(model=self.model)
            for beta in cfg.ema.betas
        }
        if self.sign_net is not None:
            self.ema_sign_nets = {
                beta: EMA(
                    model=self.sign_net,
                    beta=beta,
                    gamma=cfg.ema.gamma,
                    power=cfg.ema.power,
                )
                if beta != 1
                else EMA1(model=self.sign_net)
                for beta in cfg.ema.betas
            }
        else:
            self.ema_sign_nets = {beta: None for beta in cfg.ema.betas}

        self.all_models = {
            "model": self.model,
            "sign_net": self.sign_net,
            **{f"model_ema_{c}": m for c, m in self.ema_models.items()},
            **{f"sign_net_ema_{c}": m for c, m in self.ema_sign_nets.items()},
        }

        # checkpoint dir
        self.output_dir = Path(HydraConfig.get().runtime.output_dir)

        # Resume from checkpoint
        if cfg.training.resume:
            self.resume_from_checkpoint(cfg.training.resume)
            print(f"Resuming training from step {self.step}")
        else:
            self.step = 0
            self.best_validation_scores = {beta: -1 for beta in cfg.ema.betas}
            self.run_id = None

        # Wandb
        if cfg.wandb.logging:
            self.wandb_run = wandb.init(
                project="graph-generation",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.name,
                resume=self.run_id,
            )
            self.run_id = self.wandb_run.id
        else:
            self.run_id = None

        num_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {num_parameters / 1e6} Million")

    def save_checkpoint(self):
        checkpoint = {
            name: model.state_dict()
            for name, model in self.all_models.items()
            if model is not None
        }
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["step"] = self.step
        checkpoint["best_validation_scores"] = self.best_validation_scores
        checkpoint["run_id"] = self.run_id

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        th.save(checkpoint, checkpoint_dir / f"step_{self.step}.pt")

    def resume_from_checkpoint(self, resume):
        checkpoint_dir = self.output_dir / "checkpoints"
        assert checkpoint_dir.exists(), "No checkpoints found."
        if isinstance(resume, bool):
            # resume from latest checkpoint
            checkpoint_path = max(
                checkpoint_dir.glob("step_*.pt"),
                key=lambda f: int(f.stem.split("_")[1]),
            )
        else:
            # resume from specific checkpoint
            checkpoint_path = checkpoint_dir / f"step_{resume}.pt"

        checkpoint = th.load(checkpoint_path)
        for name, model in self.all_models.items():
            if model is not None:
                model.load_state_dict(checkpoint[name])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.best_validation_scores = checkpoint["best_validation_scores"]
        self.run_id = checkpoint["run_id"]

    def train(self):
        print(f"Training model on {self.device}")
        self.model.train()

        last_step = False
        while not last_step:
            self.step += 1
            last_step = self.step == self.cfg.training.num_steps

            step_start_time = time()
            batch = next(self.train_iterator)
            loss_terms = self.run_step(batch)
            if self.cfg.training.log_interval > 0 and (
                self.step % self.cfg.training.log_interval == 0 or last_step
            ):
                loss_terms["step_time"] = time() - step_start_time
                self.log({"training": loss_terms})

            if self.cfg.validation.interval > 0 and (
                self.step >= self.cfg.validation.first_step
                and self.step % self.cfg.validation.interval == 0
                or last_step
            ):
                if self.device == "cuda":
                    th.cuda.empty_cache()

                self.run_validation()

                if self.cfg.training.save_checkpoint:
                    self.save_checkpoint()

                if self.device == "cuda":
                    th.cuda.empty_cache()

    def test(self):
        print(f"Testing model at {self.step} steps on {self.device}")

        # Test for all EMA beta values
        test_results = {}
        for beta in self.cfg.ema.betas:
            test_results[f"ema_{beta}"] = self.evaluate(self.test_graphs, beta)

        # Log results
        self.log({"test": test_results})

        # Dump results
        if self.cfg.training.save_checkpoint:
            test_dir = self.output_dir / "test"
            test_dir.mkdir(exist_ok=True)
            with open(test_dir / f"step_{self.step}.pkl", "wb") as f:
                pickle.dump(test_results, f)

    def run_step(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        loss, loss_terms = self.method.get_loss(
            batch=batch, model=self.model, sign_net=self.sign_net
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        for model in list(self.ema_models.values()) + list(self.ema_sign_nets.values()):
            if model is not None:
                model.update(step=self.step)

        return loss_terms

    def run_validation(self):
        print(f"Running validation at {self.step} steps.")

        # Evaluate for all EMA beta values
        val_results = {}
        test_results = {}
        for beta in self.cfg.ema.betas:
            val_results[f"ema_{beta}"] = self.evaluate(self.validation_graphs, beta)

            # Compute validation score
            unique_novel_valid_keys = [
                str(m) for m in self.metrics if "UniqueNovelValid" in str(m)
            ]
            if len(unique_novel_valid_keys) > 0:
                validation_score = val_results[f"ema_{beta}"][
                    unique_novel_valid_keys[0]
                ]
            else:
                validation_score = 1 / val_results[f"ema_{beta}"]["Ratio"]

            # Evaluate on test set if validation score improved
            if validation_score >= self.best_validation_scores[beta]:
                self.best_validation_scores[beta] = validation_score
                test_results[f"ema_{beta}"] = self.evaluate(self.test_graphs, beta)

        # Log results
        self.log({"validation": val_results, "test": test_results})

        # Dump results
        if self.cfg.training.save_checkpoint:
            val_dir = self.output_dir / "validation"
            val_dir.mkdir(exist_ok=True)
            with open(val_dir / f"step_{self.step}.pkl", "wb") as f:
                pickle.dump(val_results, f)
            if test_results:
                test_dir = self.output_dir / "test"
                test_dir.mkdir(exist_ok=True)
                with open(test_dir / f"step_{self.step}.pkl", "wb") as f:
                    pickle.dump(test_results, f)

    @th.no_grad()
    def evaluate(self, eval_graphs: list[nx.Graph], beta):
        """Evaluate model for given beta on given graphs."""
        model = self.ema_models[beta]
        sign_net = self.ema_sign_nets[beta]

        # Shuffle prediction order to make size distribution more uniform
        pred_perm = self.rng.permutation(np.arange(len(eval_graphs)))

        # Select target number of nodes and split into batches
        target_size = np.array([len(g) for g in eval_graphs])[pred_perm]
        bs = (
            self.cfg.validation.batch_size
            if self.cfg.validation.batch_size is not None
            else self.cfg.training.batch_size
        )
        batches = [target_size[i : i + bs] for i in range(0, len(target_size), bs)]

        results = {}

        # Generate graphs
        pred_graphs = []
        for batch in batches:
            pred_graphs += self.method.sample_graphs(
                target_size=th.tensor(batch, device=self.device),
                model=model,
                sign_net=sign_net,
            )
        results["pred_graphs"] = [pred_graphs[i] for i in pred_perm]
        if self.device == "cuda":
            th.cuda.empty_cache()

        # Validate graphs
        for metric in self.metrics:
            results[str(metric)] = metric(
                reference_graphs=eval_graphs,
                predicted_graphs=pred_graphs,
                train_graphs=self.train_graphs,
            )

        if self.cfg.validation.per_graph_size:
            for n in set(target_size):
                eval_graphs_n = [g for g in eval_graphs if len(g) == n]
                pred_graphs_n = [g for g in pred_graphs if len(g) == n]
                results[f"size_{n}"] = {}
                for metric in self.metrics:
                    results[f"size_{n}"][str(metric)] = metric(
                        reference_graphs=eval_graphs_n,
                        predicted_graphs=pred_graphs_n,
                        train_graphs=self.train_graphs,
                    )

        # Sample plots
        n = min(4, len(self.validation_graphs)) // 2
        fig, axs = plt.subplots(n, 2, figsize=(50, 50))
        for i in range(n * n):
            G = pred_graphs[i]
            ax = axs[i // n, i % n]
            nx.draw(
                G=G,
                ax=ax,
                pos=nx.spring_layout(G, seed=42),
            )
            ax.title.set_text(f"N = {len(G)}")
            ax.title.set_fontsize(40)
        fig.tight_layout()
        results["examples"] = fig

        return results

    def log(self, log_dict: dict, prefix: str = "", indent: int = 0):
        """Logs an arbitrarily nested dict to the console and wandb."""
        for key, value in log_dict.items():
            if isinstance(value, dict):
                print(f"{'   ' * indent}{key}:")
                self.log(value, prefix=f"{prefix}{key}/", indent=indent + 1)
            elif isinstance(value, float):
                print(f"{'   ' * indent}{key}: {value}")
                if self.cfg.wandb.logging:
                    self.wandb_run.log({f"{prefix}{key}": value}, step=self.step)
            elif isinstance(value, Figure):
                if self.cfg.wandb.logging:
                    self.wandb_run.log(
                        {f"{prefix}{key}": wandb.Image(value)}, step=self.step
                    )
