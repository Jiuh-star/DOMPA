from __future__ import annotations

import copy
import typing as T
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

import flcore
import flcore.utils.model as model_utils
from attackers import get_attackers, get_attackers_id, get_benign_clients, Attacker
from system import _metrics as metrics


class FedAvgClient(flcore.Client):
    def __init__(
            self,
            *,
            id_: T.Hashable,
            device: torch.device,
            dataset_size: int,
            num_class: int,
            max_epoch: int,
            train_dataloader: D.DataLoader,
            eval_dataloader: D.DataLoader,
            test_dataloader: D.DataLoader,
            optimizer_factory: T.Callable[[T.Self], optim.Optimizer],
            loss_fn: nn.Module,
    ):
        self.id = id_
        self.device = device
        self.dataset_size = dataset_size
        self.optimizer_factory = optimizer_factory
        self.max_epoch = max_epoch
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics.MetricCollection(
            {
                "loss": metrics.MeanMetric(nan_strategy="error"),
                "accuracy": metrics.Accuracy(
                    "multiclass", num_classes=num_class, average="micro", nan_strategy="error"
                ),
            }
        ).to(self.device)

    def train(self):
        optimizer = self.optimizer_factory(self)

        self.model.train()
        for _ in range(self.max_epoch):
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _eval(self, dataloader: D.DataLoader) -> flcore.MetricResult:
        self.metrics.reset()

        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            self.metrics.update(preds=y_hat, target=y, value=loss)

        metric_result = flcore.MetricResult({name: value.item() for name, value in self.metrics.compute().items()})

        return metric_result

    def evaluate(self) -> flcore.MetricResult:
        return self._eval(self.eval_dataloader)

    def test(self) -> flcore.MetricResult:
        return self._eval(self.test_dataloader)

    def __repr__(self):
        return f"<{self.__class__.__name__} [id:{self.id},dataset:{self.dataset_size},device:{self.device}]>"


class FedAvgServer(flcore.Server):
    @torch.no_grad()
    def _eval(self, stage: T.Literal["evaluate", "test"]) -> flcore.EvaluationResult:
        """evaluate all models in benign clients"""
        client_metric_result: dict[T.Hashable, flcore.MetricResult] = {}

        for client in self.registered_clients:
            with client:
                client.receive_model(self.model)
                eval_fn = getattr(client, stage)
                client_metric_result[client.id] = eval_fn()

        # Exclude attackers
        benign_client_metric_result = {
            id_: result
            for id_, result in client_metric_result.items()
            if id_ not in get_attackers_id(self.registered_clients)
        }

        benign_collected = self._collect_evaluation_results(benign_client_metric_result)
        benign_analyzed = self._analyze_evaluation_results(benign_collected)

        # Include attackers
        collected = self._collect_evaluation_results(client_metric_result)
        analyzed = self._analyze_evaluation_results(collected)
        analyzed = {f"{k} (all)": v for k, v in analyzed.items()}

        benign_analyzed |= analyzed

        return benign_analyzed

    def __repr__(self):
        return f"<{self.__class__.__name__} [clients:{len(self.registered_clients)}]>"


class FedAvg(flcore.FederatedLearning):
    def __init__(
            self,
            *,
            server: FedAvgServer,
            log_dir: str | Path,
            tensorboard: bool = True,
            eval_per_epoch: int = 1,
            continuous: bool = False,
    ):
        super().__init__(server=server, log_dir=log_dir, tensorboard=tensorboard)
        self.eval_per_epoch = eval_per_epoch
        self.continuous = continuous
        self._saved = False

    def algorithm(self) -> nn.Module:
        self._saved = False

        self.progress.log(flcore.LogItem(
            epoch=0,
            message=f"The attackers are: {[c.id for c in get_attackers(self.server.registered_clients)]}",
        ))

        for global_epoch in self.progress(range(self.server.max_epoch), "Global Epoch"):
            # select clients to train
            selected_clients = self.server.select_clients()

            # continuous attack
            if self.continuous:
                attackers = get_attackers(self.server.registered_clients)
                benign_clients = get_benign_clients(selected_clients)
                selected_clients = (attackers + benign_clients)[:len(selected_clients)]

            self.log(flcore.LogItem(
                epoch=global_epoch,
                message=f"Selected Clients: {[c.id for c in selected_clients]}"
            ))

            self.server.connect_clients(selected_clients)

            # train
            for client in self.progress(selected_clients, "Trained Client"):
                # connect to client and train
                client.receive_model(self.server.model)
                client.train()

            # attack
            for attacker in self.progress(get_attackers(selected_clients), "Attack"):
                attacker.attack(global_epoch, selected_clients, system=self)

                if attacker.attack_log:
                    self.log(flcore.LogItem(
                        epoch=global_epoch,
                        message=f"An attacker ({attacker.id}) initiated an attack.",
                        metrics=flcore.EvaluationResult(attacker.attack_log)
                    ))
                    self.analyze(global_epoch, attacker, self.server, selected_clients)

            # aggregate
            sizes = [client.dataset_size for client in selected_clients]
            self.server.aggregate(
                models=[client.send_model() for client in selected_clients],
                weights=[size / sum(sizes) for size in sizes],
            )

            self.server.close_clients(selected_clients)

            # evaluate
            if global_epoch % self.eval_per_epoch == 0:
                metric_result = self.server.evaluate()
                self.log(flcore.LogItem(epoch=global_epoch, metrics=metric_result))

        # test the global model
        metric_result = self.server.test()
        self.log(flcore.LogItem(epoch=self.server.max_epoch,
                                metrics=metric_result,
                                others={"global_model": self.server.model}),
                 big_item=True,
                 filename="result.pth")

        return self.server.model

    def analyze(
            self,
            global_epoch: int,
            attacker: Attacker,
            server: flcore.Server,
            selected_clients: T.Sequence[flcore.Client]
    ):
        attacker_vector = model_utils.model_to_vector(attacker.model).cpu()
        global_vector = model_utils.model_to_vector(server.model).cpu()
        attacker_update = attacker_vector - global_vector
        benign_updates = [
            model_utils.model_to_vector(client.model).cpu() - global_vector
            for client in get_benign_clients(selected_clients)
        ]

        # log the l2 norm and cosine similarity with the current global model
        if self._saved is False:
            self._saved = True

            self.log(flcore.LogItem(
                epoch=global_epoch,
                metrics=flcore.EvaluationResult({
                    "L2 Norm (Attacker Model)": attacker_vector.norm().item(),
                    "L2 Norm (Global Model)": global_vector.norm().item(),

                    "L2 Norm (Attacker Update)": attacker_update.norm().item(),
                    "L2 Norms (Benign Updates)": [update.norm().item() for update in benign_updates],
                    "L2 Norm (Benign Update Max)": max([update.norm().item() for update in benign_updates]),
                    "Cosine Similarity (Attacker Update & Ref Update)": F.cosine_similarity(
                        attacker_update, torch.stack(benign_updates).mean(dim=0), dim=0,
                    ).item(),
                    "Cosine Similarity (Attacker Model & Global Model)": F.cosine_similarity(
                        attacker_vector, global_vector, dim=0
                    ).item(),
                    "Cosine Similarities (Attacker Update & Benign Updates)": [
                        F.cosine_similarity(attacker_update, update, dim=0).item()
                        for update in benign_updates
                    ]
                }),
                others={
                    # "attacker model": copy.deepcopy(attacker.model).cpu(),
                    # "global model": copy.deepcopy(self.server.model).cpu(),
                    # "server": server,
                    # "selected_clients": selected_clients,
                },
            ), big_item=True, filename=f"attack_{attacker.id}_epoch_{global_epoch}.pth")
        else:
            self.log(flcore.LogItem(
                epoch=global_epoch,
                metrics=flcore.EvaluationResult({
                    "L2 Norm (Attacker Model)": attacker_vector.norm().item(),
                    "L2 Norm (Global Model)": global_vector.norm().item(),

                    "L2 Norm (Attacker Update)": attacker_update.norm().item(),
                    "L2 Norms (Benign Updates)": [update.norm().item() for update in benign_updates],
                    "L2 Norm (Benign Update Max)": max([update.norm().item() for update in benign_updates]),
                    "Cosine Similarity (Attacker Update & Ref Update)": F.cosine_similarity(
                        attacker_update, torch.stack(benign_updates).mean(dim=0), dim=0,
                    ).item(),
                    "Cosine Similarity (Attacker Model & Global Model)": F.cosine_similarity(
                        attacker_vector, global_vector, dim=0
                    ).item(),
                    "Cosine Similarities (Attacker Update & Benign Updates)": [
                        F.cosine_similarity(attacker_update, update, dim=0).item()
                        for update in benign_updates
                    ]
                }),
                others={},
            ))

    def __repr__(self):
        return f"<{self.__class__.__name__} [server:{self.server}]>"
