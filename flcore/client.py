from __future__ import annotations

import copy
import typing as T
import warnings
from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from .utils import model as model_utils

__all__ = ["Client", "MetricResult", "Channel"]

Channel = T.NewType("Channel", T.Any)
MetricResult = T.NewType("MetricResult", dict[str, float])


class Client(ABC):
    """
    The protocol of client. A client is a device that has a local model and can train, evaluate and test the model.
    """

    id: T.Hashable
    device: torch.device
    dataset_size: int
    _context: dict
    _conn_count: int = 0

    @abstractmethod
    def train(self):
        """
        Training the local model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> MetricResult:
        """
        Evaluating the local model.

        :return: The evaluated result.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self) -> MetricResult:
        """
        Testing the local model.

        :return: The tested result.
        """
        raise NotImplementedError

    @property
    def model(self) -> nn.Module:
        """
        :return: The local model.
        """
        _context = getattr(self, "_context", None)
        if _context is None:
            raise RuntimeError("The connection is not established yet.")

        model = _context.get("model", None)
        if model is None:
            raise RuntimeError("The model is not received yet.")

        return model

    def connect(self) -> Channel:
        """
        Connect to a server.

        :return: The channel to the server.
        """
        if self._conn_count <= 0:
            self._context = {}
            self._conn_count = 0

        self._conn_count += 1

        return Channel(self)

    def close(self):
        self._conn_count -= 1

        if self._conn_count <= 0 and hasattr(self, "_context"):
            del self._context

    @property
    def connected(self) -> bool:
        return self._conn_count > 0

    def force_close(self):
        self._conn_count = 0
        self.close()

    def receive_model(self, model: nn.Module):
        """
        Receive a model.

        :param model: The new model.
        """
        if "model" in self._context:
            model_utils.move_parameters(model, self._context["model"], zero_grad=True)
        else:
            self._context["model"] = copy.deepcopy(model).to(self.device)

    def send_model(self) -> nn.Module:
        """
        Send the local model to the server.
        """
        return copy.deepcopy(self.model).cpu()

    def __enter__(self) -> Channel:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            warnings.warn(
                f"An Exception was raised: {exc_type} - {exc_val}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.close()
