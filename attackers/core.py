# TODO: Refactor for intuitive code

from __future__ import annotations

import copy
import inspect
import types
from abc import abstractmethod, ABC
from functools import cached_property
from typing import Sequence, TypeAlias, TypeVar, Iterable, Hashable

import torch
import torch.nn as nn
import torch.optim as optim

import flcore
import flcore.utils.model as model_utils

__all__ = ["BenignClient", "UnknownClient",
           "Attacker", "AttackInfo",
           "get_attackers", "get_benign_clients", "get_attackers_id", "has_attacker"]

BenignClient = TypeVar("BenignClient", bound=flcore.Client)
UnknownClient: TypeAlias = BenignClient | "Attacker"


class Attacker(flcore.Client, ABC):
    def __init__(self,
                 *,
                 client: flcore.Client,
                 attack_range: range = None,
                 attack_once: bool = False,
                 attacker_threshold: int = 0,
                 change_dataset_size_when_attack: bool = False,
                 **kwargs):
        self._client = client
        self.attack_log = {}
        self.attack_range = attack_range if isinstance(attack_range, range) else range(9999)
        self.attack_once = attack_once
        self.attacker_threshold = attacker_threshold
        self.change_dataset_size_when_attack = change_dataset_size_when_attack

        self._attacked = False

        # self._client.connect()
        self._bind_client_methods(['__getattr__', '__repr__', '__init__'])

    def attack(self, global_epoch: int, selected_clients: Sequence[flcore.Client],
               system: flcore.FederatedLearning):
        info = AttackInfo(global_epoch, selected_clients, system, self.device)
        self.attack_log.clear()

        if len(info.malicious_clients) < self.attacker_threshold:
            return

        # If the attacker has attacked, and attack_once is True, then the attacker will not attack again.
        if self.attack_once and self._attacked:
            return

        if global_epoch in self.attack_range:
            if self.change_dataset_size_when_attack:
                self.dataset_size = torch.tensor([
                    client.dataset_size for client in info.selected_clients
                ]).max().item()

            self.attack_algorithm(info)

            self._attacked = True

        # Incase memory leaks
        del info

    @abstractmethod
    def attack_algorithm(self, info: AttackInfo):
        raise NotImplementedError

    def train(self, optimizer: optim.Optimizer = None):
        """ for type checking """
        return self._client.train()

    def evaluate(self) -> flcore.MetricResult:
        """ for type checking """
        return self._client.evaluate()

    def test(self):
        """ for type checking """
        return self._client.test()

    @property
    def model(self) -> nn.Module:
        """ for type checking """
        return self._client.model

    def _optimizer_factory(self, client):
        return self.optimizer_factory(client)

    def _bind_client_methods(self, excepts: Sequence[str] = None):
        """ bind benign client methods """
        excepts = excepts or []
        client = self._client

        def _filter(name):
            if name in excepts:
                return False

            if isinstance(inspect.getattr_static(client, name), property):
                return False

            obj = getattr(client, name)
            if isinstance(obj, types.MethodType):
                return True

        client_methods = [method_name for method_name in dir(client) if _filter(method_name)]

        for client_method in client_methods:
            setattr(self, client_method, getattr(self._client, client_method))

    def __getattr__(self, item):
        """ bind benign client attributes """
        if "_client" not in vars(self):  # for pickle
            raise AttributeError
        return getattr(self._client, item)

    def __repr__(self):
        return f"<{self.__class__.__name__} [id:{self._client.id},logging:{self.attack_log},client:{self._client}]>"


def get_attackers(clients: Sequence[UnknownClient]) -> list[Attacker]:
    return [client for client in clients if isinstance(client, Attacker)]


def get_benign_clients(clients: Sequence[UnknownClient]) -> list[BenignClient]:
    return [client for client in clients if not isinstance(client, Attacker)]


def get_attackers_id(clients: Sequence[UnknownClient]) -> list[Hashable]:
    return [attacker.id for attacker in get_attackers(clients)]


def has_attacker(clients: Sequence[UnknownClient]) -> bool:
    return True if get_attackers(clients) else False


class AttackInfo:
    def __init__(self, global_epoch: int, selected_clients: Sequence[flcore.Client],
                 system: flcore.FederatedLearning, device: torch.device):
        self.global_epoch = global_epoch
        self.selected_clients = selected_clients
        self.system = system
        self.device = device

    @property
    def server(self):
        return self.system.server

    @property
    def robust_fn(self):
        return self.system.server.robust_fn

    @property
    def benign_clients(self):
        return get_benign_clients(self.selected_clients)

    @property
    def malicious_clients(self):
        return get_attackers(self.selected_clients)

    @cached_property
    def global_model(self):
        return copy.deepcopy(self.system.server.model).to(self.device)

    @property
    def global_vector(self):
        return model_utils.model_to_vector(self.global_model)

    @cached_property
    def benign_models(self):
        return [copy.deepcopy(client.model).to(self.device) for client in self.benign_clients]

    @property
    def benign_vectors(self):
        return [model_utils.model_to_vector(model) for model in self.benign_models]

    @property
    def benign_updates(self):
        return [vector - self.global_vector for vector in self.benign_vectors]

    @property
    def reference_model(self):
        return model_utils.aggregate_model(
            global_model=self.global_model,
            local_models=self.benign_models,
            weights=self.compute_weights(self.benign_clients),
        )

    @property
    def reference_vector(self):
        return model_utils.aggregate_vector(
            global_vector=self.global_vector,
            local_vectors=self.benign_vectors,
            weights=self.compute_weights(self.benign_clients),
        )

    @property
    def reference_update(self):
        return model_utils.aggregate_update(updates=self.benign_updates,
                                            weights=self.compute_weights(self.benign_clients))

    @property
    def weights(self):
        return self.compute_weights(self.selected_clients)

    @staticmethod
    def compute_weights(clients: Iterable[UnknownClient]):
        sizes = [client.dataset_size for client in clients]
        weights = [size / sum(sizes) for size in sizes]
        return weights
