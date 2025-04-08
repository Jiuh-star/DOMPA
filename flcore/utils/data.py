# REFERENCE: Federated Learning on Non-IID Data Silos: An Experimental Study
from __future__ import annotations

import math
import random
import typing as T

import more_itertools
import torch
import torch.distributions as distributions
import torch.utils.data as data

_T = T.TypeVar("_T")
Feature = T.TypeVar("Feature")
Target = T.TypeVar("Target")
Dataset: T.TypeAlias = data.Dataset


def default_get_target(item: tuple[T.Any, Target]) -> Target:
    """
    Default function that receives an item from dataset, and then return its target.

    :param item: An item from dataset.
    :return: Target of the item.
    """
    return item[-1].tolist() if isinstance(item[-1], torch.Tensor) else item[-1]


def get_targets(
        dataset: Dataset, get_target: T.Callable[[T.Any], Target] = default_get_target
) -> list[Target]:
    """
    Get targets from dataset.

    :param dataset: The dataset.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of targets of dataset.
    """
    # dataset from torchvision may have attribute targets
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        # some targets are Tensor type
        targets = targets.tolist() if isinstance(targets, torch.Tensor) else targets

        return list(targets)

    # some user may concat torchvision dataset
    if isinstance(dataset, data.ConcatDataset):
        targets = []
        for _dataset in dataset.datasets:
            targets.extend(get_targets(_dataset))
        return targets

    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        targets = [targets[index] for index in dataset.indices]
        return targets

    targets = [get_target(item) for item in dataset]
    setattr(dataset, "targets", targets)

    return targets


def collect_targets(targets: T.Sequence[Target]) -> dict[Target, list[Target]]:
    """
    Collect and classify the targets to a dict. In dict, the keys are targets and value are corresponding indexes.

    :param targets: A sequence of target.
    :return: Collected targets.
    """
    target_indexes = {}
    for index, target in enumerate(targets):
        target_indexes.setdefault(target, []).append(index)
    return target_indexes


def random_split(
        values: T.Sequence[_T],
        lengths: T.Optional[T.Sequence[int]] = None,
        fractions: T.Optional[T.Sequence[float]] = None,
) -> list[list[_T]]:
    """
    Randomly split *values* into some list of value.

    :param values: A Sequence of values to be split.
    :param lengths: Lengths of subsequence. Should equals to len(values).
    :param fractions: Fractions of subsequence length. Should be closed to 1 enough.
    :return: A list of sublist of values.

    :raises ValueError: If lengths or fractions are not legal.
    """
    if lengths and fractions:
        raise ValueError(
            "Both lengths and fractions are specified, consider remove one."
        )

    if not lengths and not fractions:
        raise ValueError("No values in lengths and fractions at the same time.")

    if lengths and sum(lengths) != len(values):
        raise ValueError("Sum of lengths is not equals to values length.")

    if fractions:
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-5):
            raise ValueError("Sum of fractions is not close enough to 1.")

        # fractions to lengths
        lengths = []
        for i, frac in enumerate(fractions):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            length = math.floor(len(values) * frac)
            lengths.append(length)

        # sum(lengths) may not exactly equals to len(values)
        remainder = len(values) - sum(lengths)
        remainder = remainder if remainder >= 0 else -remainder
        step = 1 if remainder >= 0 else -1

        # put remainder into lengths
        for i in range(remainder):
            lengths[i % len(lengths)] += step

    values = list(values)
    random.shuffle(values)
    values = list(more_itertools.split_into(values, lengths))  # type: ignore

    return values  # type: ignore


def partition_dataset(
        dataset: Dataset,
        num_partition: int,
        partition_func: T.Callable[[], list[float]],
        get_target: T.Callable[[T.Any], T.Any] = default_get_target,
) -> list[data.Subset]:
    """
    Partition dataset into *num_partition* parts, the lengths of the parts follow the return value of *partition_func*.

    :param dataset: The dataset.
    :param num_partition: The quantity of partition.
    :param partition_func: A callable that return the partition of each class. The length of return should be equal to
     num_partition.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of subsets.

    :raises ValueError: If the length of partition_func didn't return the right fractions.
    """
    targets = get_targets(dataset, get_target)
    target_indexes = collect_targets(targets)
    target_split_indexes = {}

    for target in target_indexes:
        fractions = partition_func()
        if len(fractions) != num_partition:
            raise ValueError(
                f"The length of partition_func() return ({len(fractions)}) "
                f"is not equal to num_partition ({num_partition})."
            )

        target_split_indexes[target] = random_split(
            target_indexes[target], fractions=fractions
        )

    # make subsets
    subsets = []
    for _ in range(num_partition):  # for each partition
        indexes = []
        for (
                target
        ) in target_split_indexes:  # put split indexes of each class to a partition
            indexes.extend(target_split_indexes[target].pop())

        subset = data.Subset(dataset, indexes)
        subsets.append(subset)

    return subsets


def generate_dirichlet_subsets(
        dataset: Dataset,
        alphas: T.Sequence[float],
        get_target: T.Callable[[T.Any], T.Any] = default_get_target,
        min_data: int = 10,
        max_retry: int = 10,
) -> list[data.Subset]:
    """
    Generate subsets that follow dirichlet distribution.

    :param dataset: The source dataset.
    :param alphas: The parameter of dirichlet distribution.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :param min_data: The minimal dataset size of each subset.
    :param max_retry: Max retry of sample from dirichlet distribution.
    :return: The subsets that follow dirichlet distribution.

    :raises TimeoutError: If unable to sample from dirichlet distribution within max_retry retries.
    """

    def dirichlet() -> list[float]:
        m = distributions.Dirichlet(torch.tensor(alphas, dtype=torch.float))
        sample = m.sample().tolist()
        return sample

    # verify min_data was satisfied
    for _ in range(max_retry):
        subsets = partition_dataset(
            dataset,
            num_partition=len(alphas),
            partition_func=dirichlet,
            get_target=get_target,
        )
        if all([len(subset) >= min_data for subset in subsets]):
            break
    else:
        raise TimeoutError(
            f"Unable to sample from dirichlet distributions (α = {alphas}) within {max_retry} retries "
            f"to satisfy that each client holds {min_data} data at least."
        )

    return subsets


def generate_p_degree_subsets(
        dataset: Dataset,
        p: float,
        num_partition: int,
        get_target: T.Callable[[T.Any], T.Any] = default_get_target,
) -> list[data.Subset]:
    """
    Generate subsets that follow p degree of non-IID. See *Local Model Poisoning Attacks to Byzantine Robust Federated
    Learning*.

    :param dataset: The source dataset.
    :param p: The parameter of p degree of non-IID.
    :param num_partition: The quantity of partition, namely the number of clients.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of subsets.
    """
    assert 0 < p < 1

    called = 0
    group_size = len(set(get_targets(dataset)))

    def p_degree() -> list[float]:
        nonlocal called

        lth_group = [p / group_size] * group_size
        no_lth_group = [(1 - p) / (num_partition - group_size)] * (
                num_partition - group_size
        )
        fractions = (
                no_lth_group[: called * group_size]
                + lth_group
                + no_lth_group[called * group_size:]
        )
        called += 1

        return fractions

    return partition_dataset(
        dataset,
        num_partition=num_partition,
        partition_func=p_degree,
        get_target=get_target,
    )


def generate_iid_subsets(
        dataset: Dataset,
        num_partition: int,
        get_target: T.Callable[[T.Any], T.Any] = default_get_target,
) -> list[data.Subset]:
    """
    Generate subsets that follow IID.

    :param dataset: The source dataset.
    :param num_partition: The quantity of partition, namely the number of clients.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: An IID subsets.
    """
    return partition_dataset(
        dataset,
        num_partition=num_partition,
        partition_func=lambda: [1 / num_partition] * num_partition,
        get_target=get_target,
    )
