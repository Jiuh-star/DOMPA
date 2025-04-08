from __future__ import annotations

import math
import random
import typing as T

import torch
import torch.nn as nn

from .client import Client, MetricResult
from .utils import model as model_utils
from .utils.robust import RobustFn

EvaluationResult = T.NewType(
    "EvaluationResult", dict[str, MetricResult | dict[str, MetricResult]]
)


class Server:
    def __init__(
            self,
            *,
            model: nn.Module,
            select_ratio: float,
            max_epoch: int,
            learning_rate: float = 1.0,
            robust_fn: T.Optional[RobustFn] = None,
    ):
        """
        The server in federated learning. It works with some clients in ``FederatedLearning``.

        You can specify global learning rate in `learning_rate`. The global learning rate may distinctly slow down the
        convergence time but take smoother loss variance in return.

        :param select_ratio: Client select ratio of each round in federated learning.
        :param max_epoch: The max epoch of server, namely global rounds of federated learning.
        :param learning_rate: The global learning rate, this works in aggregation.
        :param robust_fn: The robust aggregation function.
        """
        self.model = model
        self.select_ratio = select_ratio
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.robust_fn = robust_fn
        self.registered_clients: list[Client] = []
        self._pool = []

    def register_client(self, client: Client):
        """
        Register a client to server.

        :param client: Client to be registered.

        :raises ValueError: When client is not an available client.
        :raises RuntimeError: When there are duplicated client id.
        """
        if not isinstance(client, Client):
            raise ValueError(f"The client is not an available client.")

        if self.get_client(id_=client.id):
            raise RuntimeError(f'Conflicted client id "{client.id}"')

        self.registered_clients.append(client)

    def unregister_client(self, id_: T.Hashable) -> Client | None:
        """
        Unregister a client in server, return this client.

        :param id_: Client id
        :return: A client or None if no this client
        """
        hashed_id = hash(id_)
        for client in self.registered_clients:
            if hash(client.id) == hashed_id:
                return self.registered_clients.pop()
        return None

    def get_client(self, id_: T.Hashable) -> Client | None:
        """
        Get the registered client of `id`.

        :param id_: Client's id
        :return: The client, or None if not found
        """
        hashed_id = hash(id_)
        for client in self.registered_clients:
            if hash(client.id) == hashed_id:
                return client
        return None

    def select_clients(self) -> list[Client]:
        """
        Randomly select ``int(select_ratio * num_registered_clients)`` clients.

        :return: Selected clients
        """
        num_select = int(len(self.registered_clients) * self.select_ratio)
        selected_clients = random.sample(self.registered_clients, k=num_select)
        return selected_clients

    @staticmethod
    def connect_clients(clients: T.Iterable[Client]):
        """
        Connect clients to server.

        :param clients: The clients to be connected.
        """
        for client in clients:
            client.connect()

    @staticmethod
    def close_clients(clients: T.Iterable[Client]):
        """
        Close clients.

        :param clients: The clients to be closed.
        """
        for client in clients:
            client.close()

    def aggregate(self, models: T.Sequence[nn.Module], weights: T.Sequence[float]):
        """
        Aggregate local models to global model by aggregating updates (delta of models). Each model update will be
        multiplied by server's learning rate and corresponding weight to perform aggregation.

        :param models: Models to be aggregated.
        :param weights: Weight that corresponding model update, expected the sum equals to 1.

        :raises ValueError: When no models to be aggregated.
        :raises ValueError: When the length of weights and models are not the same.
        :raises ValueError: When sum of weights is not 1.
        """
        if len(models) == 0:
            raise ValueError("Not enough models to perform aggregation.")

        if len(weights) != len(models):
            raise ValueError(
                f"The length of weights ({len(weights)}) and clients ({len(models)}) are not the same."
            )

        if not math.isclose(math.fsum(weights), 1.0):
            raise ValueError(
                f"The sum of weights should be closed to 1, got {sum(weights)}."
            )

        if self.robust_fn:
            models, weights = self.robust_fn(self.model, models, weights)

        weights = [self.learning_rate * weight for weight in weights]
        self.model = model_utils.aggregate_model(self.model, models, weights)

    def evaluate(self) -> EvaluationResult:
        """
        Evaluate global model on all registered clients and compute mean and std for each metric.

        :return: Mean, std and raw values of client evaluation result of each metric.
        """
        return self._eval(stage="evaluate")

    def test(self) -> EvaluationResult:
        """
        Test global model on all registered clients and compute mean and std for each metric.

        :return: Mean, std and raw values of client test result of each metric.
        """
        return self._eval(stage="test")

    @torch.no_grad()
    def _eval(self, stage: T.Literal["evaluate", "test"]) -> EvaluationResult:
        """evaluate all models in clients"""
        client_metric_result: dict[T.Hashable, MetricResult] = {}

        for i, client in enumerate(self.registered_clients):
            with client:
                client.receive_model(self.model)
                eval_fn = getattr(client, stage)
                client_metric_result[client.id] = eval_fn()

        metric_client_result = self._collect_evaluation_results(client_metric_result)
        evaluation_result = self._analyze_evaluation_results(metric_client_result)

        return evaluation_result

    @staticmethod
    def _collect_evaluation_results(
            client_metric_result: dict[T.Hashable, MetricResult]
    ) -> dict[str, dict[T.Hashable, float]]:
        """
        Rotate a recursive dict ``{client: {metric: result}}`` to ``{metric: {client: result}}``.
        """
        # rotate client_metric_result
        metric_client_result = {}
        for client, metric_result in client_metric_result.items():
            for metric, result in metric_result.items():
                metric_client_result.setdefault(metric, {})[client] = result

        return metric_client_result

    @staticmethod
    def _analyze_evaluation_results(
            metric_client_result: dict[str, dict[T.Hashable, float]]
    ) -> EvaluationResult:
        """
        Compute mean and std of each metric results.
        """
        evaluation_result = {}
        for metric, client_result in metric_client_result.items():
            results = torch.tensor(list(client_result.values()))
            evaluation_result[f"{metric} (mean)"] = results.mean().item()
            evaluation_result[f"{metric} (std)"] = results.std(None).item()
            evaluation_result[f"{metric} (raw)"] = client_result
        return EvaluationResult(evaluation_result)
