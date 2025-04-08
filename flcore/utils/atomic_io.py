from __future__ import annotations

import contextlib
import pathlib
from typing import Any

import torch


def dump(obj: Any, filename: str | pathlib.Path, *, replace: bool = False):
    """
    Dump object to file.

    :param obj: The object to dump.
    :param filename: The filename.
    :param replace: Replace the file if it exists.

    :raise FileExistsError: If file exists and replace is False.
    """
    filename = pathlib.Path(filename)
    temp_file = filename.with_suffix(".tmp")
    torch.save(obj, temp_file)

    if replace:
        temp_file.replace(filename)
    else:
        temp_file.rename(filename)


def load(filename: str | pathlib.Path, *, raise_error: bool = True) -> Any:
    """
    Load object from file.

    :param filename: The filename.
    :param raise_error: Raise error if file not exists.
    :return: The object.
    """
    cm = contextlib.nullcontext() if raise_error else contextlib.suppress(Exception)

    with cm:
        return torch.load(filename)
