from __future__ import annotations

import dataclasses
import pathlib
import time
import typing as T
from abc import ABC, abstractmethod
from numbers import Number

import rich.progress
import rich.repr
import rich.table
import rich.text
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from .server import EvaluationResult, Server
from .utils import atomic_io

_T = T.TypeVar("_T")


@dataclasses.dataclass
class LogItem:
    """
    The log item of a federated learning system.
    """
    epoch: int
    message: str = ""
    metrics: EvaluationResult = dataclasses.field(default_factory=lambda: EvaluationResult({}))
    others: dict = dataclasses.field(default_factory=dict)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.epoch
        yield "message", self.message, ""
        # we only log scalar metrics in terminal to make it clear
        yield "metrics", {name: value for name, value in self.metrics.items() if isinstance(value, Number)}, {}
        yield "other", self.others, {}


class _SpeedColumn(rich.progress.ProgressColumn):
    def render(self, task: rich.progress.Task) -> rich.text.Text:
        """ Show step-style speed. """
        speed = task.finished_speed or task.speed
        if speed is None:
            return rich.text.Text("?", style="progress.data.speed")
        return rich.text.Text(f"{speed:.2f} step/s", style="progress.data.speed")


class _Progress(rich.progress.Progress):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("speed_estimate_period", 60 * 10)  # 10 minutes
        super().__init__(*args, **kwargs)
        self.columns = (
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(bar_width=None),
            _SpeedColumn(),
            rich.progress.MofNCompleteColumn(table_column=rich.table.Column(justify="center")),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
        )
        self.header_task = {}

    def __call__(self, sequence: T.Sequence[_T], header: str, **kwargs) -> T.Iterator[_T]:
        """
        Track the `sequence` and print progress bar in terminal with title `header`.

        :param sequence: Sequence to be iterated.
        :param header: Title of progress bar.
        :param kwargs: Other keyword arguments in rich.progress.Progress.track()
        :return: The element in sequence.
        """
        if not sequence:
            return sequence

        if header not in self.header_task:
            self.header_task[header] = self.add_task(header)

        task_id = self.header_task[header]
        with self._lock:
            task = self.tasks[task_id]
            task.completed = 0
            task.finished_time = None
            task.finished_speed = None

        yield from super().track(sequence, task_id=task_id, **kwargs)


class FederatedLearning(ABC):
    def __init__(self, *, server: Server, log_dir: str | pathlib.Path, tensorboard: bool = True):
        """
        The base class of a simulated federated learning system.

        :param server: A federated learning server.
        :param log_dir: The log directory of logging file.
        :param tensorboard: Logging evaluation metrics to tensorboard.
        """
        self.server = server
        self.log_dir = pathlib.Path(log_dir)
        self.tensorboard = SummaryWriter(log_dir) if tensorboard else None
        self.logbook: list[LogItem] = []
        self.progress = _Progress()

        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def algorithm(self) -> nn.Module:
        """
        The abstract method of federated learning algorithm.

        :return: The trained model.
        """

    def run(self):
        """
        Run the simulated federated learning algorithm.

        :return: The trained model.
        """
        if len(self.server.registered_clients) == 0:
            raise ValueError("There are no registered clients in server.")

        try:
            with self.progress:
                model = self.algorithm()

        except KeyboardInterrupt:
            epoch = self.logbook[-1].epoch if self.logbook else -1
            if epoch >= 0:
                self.progress.log("Interruption detected, saving the server and clients.", style="red bold")

                # try to safely close all clients
                for client in self.server.registered_clients:
                    client.close()

                    if client.connected:
                        client.force_close()

                epoch = self.logbook[-1].epoch

                atomic_io.dump(self.server, self.log_dir / f"server-{epoch}.ckpt", replace=True)

            exit(0)

        return model

    def log(self, log_item: LogItem, *, big_item: bool = False, filename: T.Optional[str] = None):
        """
        Log the *log_item* to terminal and file (`self.log_dir`).

        :param log_item: An LogItem instance.
        :param big_item: Save the log item into an independent file.
        :param filename: The filename of big_item
        """
        self.progress.log(log_item, _stack_offset=2)
        self.logbook.append(log_item)

        # Save big log_item into an independent file.
        if big_item:
            self.logbook.pop()
            filename = filename or f"log-{log_item.epoch}-{time.strftime('%X')}.pth"
            atomic_io.dump(log_item, self.log_dir / filename, replace=True)

        # Plot metric in tensorboard
        if self.tensorboard:
            for metric, value in log_item.metrics.items():
                if isinstance(value, (int, float)):  # log scalar
                    self.tensorboard.add_scalar(metric, value, global_step=log_item.epoch)
                elif isinstance(value, dict):  # we don't log raw metric since its large quantity
                    pass

        # Save the new log_item into logbook.pth.
        # Note that torch.save doesn't support incremental save, we save the whole logbook instead,
        # so do not save any big log_item into logbook.
        atomic_io.dump(self.logbook, self.log_dir / "logbook.pth", replace=True)
