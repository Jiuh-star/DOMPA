"""
Durability-Optimized Model Poisoning Attack
"""
from __future__ import annotations

import copy
import itertools
from typing import Literal, TYPE_CHECKING

import torch
import torch.nn as nn

import flcore.utils.model as model_utils

from attackers.core import Attacker, AttackInfo

if TYPE_CHECKING:
    from system.fedavg import FedAvgClient

__all__ = ["DompaAttacker"]


class DompaAttacker(Attacker):
    _clients: list[FedAvgClient] = []
    _bad_model: nn.Module = None
    _attack_at: int = -1
    _state_dict: dict = {}

    def __init__(
        self,
        *,
        client: FedAvgClient,
        learning_rate: float,
        max_epoch: int,
        momentum: float,
        max_scale: float,
        method: Literal["norm", "param", "param norm"],
        max_loss: float,
        **kwargs,
    ):
        super().__init__(client=client, **kwargs)
        self._lr = learning_rate
        self._max_epoch = max_epoch
        self._momentum = momentum
        self._max_scale = max_scale
        self._method = method
        self._max_loss = max_loss
        self._norm_supremum = -1
        self._norm_infimum = -1
        self.__class__._clients.append(client)
        client.connect()  # Keep client alive

        self._buffer: nn.Module | None = None

    def attack_algorithm(self, info: AttackInfo):
        # To speed up attack algorithm
        if self._attack_at == info.global_epoch:
            model_utils.move_parameters(self._bad_model.to(self.device), self.model)
            return

        # Forget learned model
        self.receive_model(info.global_model)

        # Get an optimizer for attacking
        optimizer_type = torch.optim.SGD
        optimizer = optimizer_type(self.model.parameters(), lr=self._lr, momentum=self._momentum)

        # Momentum between attacks
        if self._state_dict:
            optimizer.load_state_dict(self._state_dict)

        for i in range(self._max_epoch):
            losses = []
            self._reset_constrain_param(info)

            for x, y in self._data_iter():
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                # Incase some of the losses growing up too fast
                if loss > self._max_loss:
                    print("Constrain Loss:", loss.item())
                    loss = torch.log(loss)

                # Gradient ascend
                loss = -loss
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # Constraint malicious momentum update
                vector = model_utils.model_to_vector(self.model)
                update = vector - info.global_vector

                update = self._infimum_constrain(info, update)  # Infimum constrain to enhance bad update performance
                update = self._supremum_constrain(
                    info, update
                )  # Supremum to make sure final malicious update update can bypass robust algo

                vector = info.global_vector + update
                model_utils.vector_to_model(vector, self.model)

            # For analyze
            if losses:
                self.attack_log["loss"] = sum(losses) / len(losses)
                self.attack_log["loss (min)"] = min(losses)
                self.attack_log["loss (max)"] = max(losses)

            print(
                "Epoch:",
                i,
                "Loss:",
                abs(self.attack_log["loss"]),
                "Loss (min):",
                abs(self.attack_log["loss (min)"]),
                "Loss (max):",
                abs(self.attack_log["loss (max)"]),
                "Norm:",
                (model_utils.model_to_vector(self.model) - info.global_vector).norm().item(),
                "Scale:",
                self.attack_log.get("scale_factor", -1),
                "Deviation:",
                self.attack_log.get("deviation", -1),
                "Infimum:",
                self._norm_infimum,
                "Supremum:",
                self._norm_supremum,
            )

        # Build final malicious model
        self._reset_constrain_param(info)

        vector = model_utils.model_to_vector(self.model)
        update = vector - info.global_vector
        update = update - info.reference_update
        update = self._supremum_constrain(info, update)

        vector = info.global_vector + update
        model_utils.vector_to_model(vector, self.model)

        self.__class__._attack_at = info.global_epoch
        self.__class__._bad_model = copy.deepcopy(self.model)
        self.__class__._state_dict = optimizer.state_dict()

        self.attack_log["lr"] = self._lr
        self.attack_log["epoch"] = self._max_epoch
        self.attack_log["max_scale"] = self._max_scale

    def _reset_constrain_param(self, info: AttackInfo):
        if "param" in self._method:
            self._reset_param_constrain_param(info)
        if "norm" in self._method:
            self._reset_norm_constrain_param(info)

    def _infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        if self._method == "param":
            update = self._param_infimum_constrain(info, update)
        elif self._method == "norm":
            update = self._norm_infimum_constrain(info, update)
        elif self._method == "param norm":
            update = self._param_infimum_constrain(info, update)
            update = self._norm_infimum_constrain(info, update)
        else:
            raise ValueError(f"Unknown method {self.method}")
        return update

    def _supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        if self._method == "param":
            update = self._param_supremum_constrain(info, update)
        elif self._method == "norm":
            update = self._norm_supremum_constrain(info, update)
        elif self._method == "param norm":
            update = self._param_supremum_constrain(info, update)
            update = self._norm_supremum_constrain(info, update)
        else:
            raise ValueError(f"Unknown method {self.method}")

        return update

    def _reset_param_constrain_param(self, info: AttackInfo, **kwargs):
        benign_updates = torch.stack(info.benign_updates)
        self._supremum_max_boundary = benign_updates.max(dim=0)[0]
        self._supremum_min_boundary = benign_updates.min(dim=0)[0]
        self._infimum_max_boundary = self._supremum_max_boundary * 0.85
        self._infimum_min_boundary = self._supremum_min_boundary * 0.85

    def _param_infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        direction = update - info.reference_update

        block = torch.logical_and(update > self._infimum_min_boundary, update < self._infimum_max_boundary)

        mask = torch.logical_and(direction > 0, block)
        update[mask] = self._infimum_max_boundary[mask]

        mask = torch.logical_and(direction < 0, block)
        update[mask] = self._infimum_min_boundary[mask]

        return update

    def _param_supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        mask = update > self._supremum_max_boundary
        update[mask] = self._supremum_max_boundary[mask]

        mask = update < self._supremum_min_boundary
        update[mask] = self._supremum_min_boundary[mask]

        return update

    def _reset_norm_constrain_param(self, info: AttackInfo):
        self._norm_infimum = -1
        self._norm_supremum = -1

    def _norm_supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        norm_infimum, norm_supremum = self._find_norm_robust_boundary(info, update)
        update_norm = update.norm().item()

        # Incase Infimum > Supremum
        if norm_infimum > norm_supremum:
            norm_supremum = norm_infimum

        # Supremum
        if update_norm > norm_supremum:
            update = update / update_norm * norm_supremum

        return update

    def _norm_infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        norm_infimum, norm_supremum = self._find_norm_robust_boundary(info, update)
        update_norm = update.norm().item()

        # Infimum
        if update_norm < norm_infimum:
            update = update / update_norm * norm_infimum

        return update

    def _find_norm_robust_boundary(self, info: AttackInfo, update: torch.Tensor):
        if self._norm_infimum > 0 and self._norm_supremum > 0:
            return self._norm_infimum, self._norm_supremum

        # We don't have to calculate boundary every time, change lr to a smaller value may be better and quicker
        buffer = copy.deepcopy(self.model)
        update = update / update.norm() * torch.stack(info.benign_updates).norm(dim=1).mean()
        max_deviation = -1
        max_scale_factor = 1

        # Find the robust boundary
        for scale_factor in itertools.chain(
            torch.arange(0.001, 1, (1 - 0.001) / 50),
            torch.arange(1, self._max_scale, (self._max_scale - 1) / 100),
        ):
            # If there are no robust function, we can do whatever we want
            if not info.robust_fn:
                max_scale_factor = self._max_scale
                break

            scaled_vector = info.global_vector + scale_factor * update

            # evaluate this scaled model can or cannot be aggregated into global model
            model_utils.vector_to_model(scaled_vector, buffer)
            local_models = list(info.benign_models) + ([buffer] * len(info.malicious_clients))
            robust_models, weights = info.robust_fn(
                global_model=info.global_model,
                local_models=local_models,
                # attack should not know the weights of benign clients
                weights=[1 / len(local_models)] * len(local_models),
            )
            agg_vector = model_utils.aggregate_vector(
                info.global_vector,
                [model_utils.model_to_vector(model) for model in robust_models],
                weights,
            )
            agg_update = agg_vector - info.global_vector
            deviation = (agg_update.dist(info.reference_update) ** 2).item()
            # We aim to move model to step away from final global model (~=reference_model)
            # deviation = (agg_vector.dist(info.reference_vector) ** 2).item()

            # if this scaled update cause a larger deviation than before, remember it
            if deviation > max_deviation:
                max_deviation = deviation
                max_scale_factor = scale_factor.item()

        # After finding the robust boundary, we know the infimum and supremum
        self._norm_supremum = (max_scale_factor * update).norm().item()
        self._norm_infimum = max(info.reference_update.norm().item(), self._norm_supremum * 0.5)
        self._scale_factor = max_scale_factor

        self.attack_log["scale_factor"] = max_scale_factor
        self.attack_log["deviation"] = max_deviation
        self.attack_log["infimum"] = self._norm_infimum
        self.attack_log["supremum"] = self._norm_supremum

        return self._norm_infimum, self._norm_supremum

    def _data_iter(self):
        for client in self._clients:
            for x, y in client.train_dataloader:
                yield x, y
