from __future__ import annotations

import json
import random
import typing as T
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
import torch.utils.data as D
import torchvision as tv
import torchvision.transforms.functional as tvf

import flcore.utils.data as data_utils


@dataclass
class ClientDataInfo:
    id: T.Hashable
    train_dataset: D.Dataset
    eval_dataset: D.Dataset
    test_dataset: D.Dataset

    @staticmethod
    def get_dataset_size(dataset) -> int:
        if __len__ := getattr(dataset, "__len__", None):
            return __len__()
        return 1

    @cached_property
    def train_dataset_size(self) -> int:
        return self.get_dataset_size(self.train_dataset)

    @cached_property
    def eval_dataset_size(self) -> int:
        return self.get_dataset_size(self.eval_dataset)

    @cached_property
    def test_dataset_size(self) -> int:
        return self.get_dataset_size(self.test_dataset)

    @cached_property
    def dataset_size(self) -> int:
        return self.train_dataset_size + self.eval_dataset_size + self.test_dataset_size


@dataclass
class FlDataset:
    client_data: dict[T.Hashable, ClientDataInfo]
    fractions: tuple[int, int, int]
    others: dict

    def plot(self):
        raise NotImplementedError


def _tv_dataset(
        dataset: D.Dataset,
        num_client: int,
        min_data: int = 40,
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2),
        alpha: float = None,
        p_degree: float = None,
        num_class: int = 10,
) -> FlDataset:
    if p_degree is not None:
        subsets = data_utils.generate_p_degree_subsets(dataset, p_degree, num_partition=num_client)
    else:
        subsets = data_utils.generate_dirichlet_subsets(
            dataset,
            alphas=[alpha] * num_client,
            min_data=min_data,
            max_retry=100,
        )

    client_data = _split_into_client_data(subsets, fractions)

    fl_dataset = FlDataset(client_data, fractions, others={"num_class": num_class})

    return fl_dataset


def cifar10(
        num_client: int,
        resize: tuple[int, int] = (32, 32),
        min_data: int = 40,
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2),
        alpha: float = None,
        p_degree: float = None,
) -> FlDataset:
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(resize, antialias=True),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = D.ConcatDataset([
        tv.datasets.CIFAR10(
            "data", download=True, train=True, transform=transforms
        ),
        tv.datasets.CIFAR10(
            "data", download=True, train=False, transform=transforms
        ),
    ])
    return _tv_dataset(dataset, num_client=num_client, min_data=min_data, fractions=fractions, alpha=alpha,
                       p_degree=p_degree)


def fashionmnist(
        num_client: int,
        resize: tuple[int, int] = (32, 32),
        min_data: int = 40,
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2),
        alpha: float = None,
        p_degree: float = None,
) -> FlDataset:
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(resize, antialias=True),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = D.ConcatDataset([
        tv.datasets.FashionMNIST(
            "data", download=True, train=True, transform=transforms
        ),
        tv.datasets.FashionMNIST(
            "data", download=True, train=False, transform=transforms
        ),
    ])
    return _tv_dataset(dataset, num_client=num_client, min_data=min_data, fractions=fractions, alpha=alpha,
                       p_degree=p_degree)


def mnist(
        num_client: int,
        resize: tuple[int, int] = (32, 32),
        min_data: int = 40,
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2),
        alpha: float = None,
        p_degree: float = None,
) -> FlDataset:
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(resize, antialias=True),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = D.ConcatDataset([
        tv.datasets.MNIST(
            "data", download=True, train=True, transform=transforms
        ),
        tv.datasets.MNIST(
            "data", download=True, train=False, transform=transforms
        ),
    ])
    return _tv_dataset(dataset, num_client=num_client, min_data=min_data, fractions=fractions, alpha=alpha,
                       p_degree=p_degree)


def femnist(
        num_client: int = -1,
        resize: list[int] = (32, 32),
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2)
) -> FlDataset:
    root = Path("data/femnist.pth")
    subsets = torch.load(root)

    if num_client > 0:
        subsets = random.choices(subsets, k=num_client)

    for subset in subsets:
        subset.tensors = (
            tvf.resize(subset.tensors[0], resize, antialias=True),
            subset.tensors[1],
        )

    client_data = _split_into_client_data(subsets, fractions)

    fl_dataset = FlDataset(client_data, fractions, others={"num_class": 62})

    return fl_dataset


def shakespeare(
        num_client: int = -1,
        fractions: tuple[int, int, int] = (0.6, 0.2, 0.2)
) -> FlDataset:
    # load data
    root = Path("data/shakespeare.json")
    with open(root) as f:
        raw_data = json.load(f)

    # limit number of clients
    if num_client > 0:
        raw_data["users"] = random.choices(raw_data["users"], k=num_client)

    # data preprocess
    counter = Counter(char
                      for item in raw_data["user_data"].values()
                      for char in item["y"])
    counter.update(char
                   for item in raw_data["user_data"].values()
                   for words in item["x"]
                   for char in words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    char2token = {char: i for i, char in enumerate(vocab, 1)}
    token2char = {i: char for i, char in enumerate(vocab, 1)}

    # build fl dataset
    client_data = {}
    for user in raw_data["users"]:
        dataset = ShakespeareCharacterDataset(raw_data["user_data"][user], char2token, token2char)

        indices = list(range(len(dataset)))
        train_size = int(len(dataset) * fractions[0])
        eval_size = int(len(dataset) * fractions[1])

        train_set = D.Subset(dataset, indices[:train_size])
        eval_set = D.Subset(dataset, indices[train_size:train_size + eval_size])
        test_set = D.Subset(dataset, indices[train_size + eval_size:])

        client_data[user] = ClientDataInfo(
            id=user,
            train_dataset=train_set,
            eval_dataset=eval_set,
            test_dataset=test_set,
        )

    # save fl dataset
    fl_dataset = FlDataset(client_data, fractions, others={"num_class": len(vocab) + 1})

    return fl_dataset


def _split_into_client_data(subsets, fractions):
    client_data = {}
    for i, subset in enumerate(subsets):
        train_set, eval_set, test_set, *_ = D.random_split(subset, fractions)

        client_data[i] = ClientDataInfo(
            id=i,
            train_dataset=train_set,
            eval_dataset=eval_set,
            test_dataset=test_set,
        )
    return client_data


class ShakespeareCharacterDataset(D.Dataset):
    def __init__(self, data, char2token, token2char):
        self.data = data
        self.char2token = char2token
        self.token2char = token2char

    def __getitem__(self, item):
        return self.encode(self.data["x"][item]), self.encode(self.data["y"][item])

    def __len__(self):
        return len(self.data["x"])

    def encode(self, x: str) -> torch.Tensor:
        if len(x) == 1:
            return self.char2token[x]

        return torch.tensor([self.char2token[c] for c in x])

    def decode(self, x: torch.Tensor) -> str:
        if x.size() == torch.Size([]):
            return self.token2char[x.item()]

        return "".join(self.token2char[i.item()] for i in x)
