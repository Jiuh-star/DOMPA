from __future__ import annotations

import copy
import typing as T

import torch
import torch.nn as nn


@torch.no_grad()
def move_parameters(
        from_: nn.Module, to: nn.Module, *, buffer: bool = False, zero_grad: bool = False
):
    """
    Move parameters from one model to another.

    :param from_: The model to move parameters from.
    :param to: The model to move parameters to.
    :param buffer: Whether to move buffers.
    :param zero_grad: Set the gradient of the parameters to None.
    """
    if buffer:
        to.load_state_dict(from_.state_dict())
    else:
        for from_param, to_param in zip(from_.parameters(), to.parameters(), strict=True):
            device = to_param.device
            to_param.data.copy_(from_param.data.to(device))

    if zero_grad:
        to.zero_grad(set_to_none=True)


@torch.no_grad()
def model_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Convert model parameters to a detached vector.

    :param model: The model.
    :return: The vector.
    """
    # Flag for the device where the parameter is located
    param_device = None

    vector = []
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vector.append(param.view(-1))

    return torch.cat(vector)


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    """
    Convert a vector to model parameters, and copy the data to the model.

    :param vector: The vector.
    :param model: The model.

    :raise TypeError: If the vector is not a Tensor.
    """
    # Ensure vec of type Tensor
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got: {torch.typename(vector)}")

    # Flag for the device where the parameter is located
    param_device = None
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()

        # Slice the vector, reshape it, and copy the old data of the parameter
        param.data.copy_(vector[pointer: pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param


@torch.no_grad()
def aggregate_update(
        updates: T.Iterable[torch.Tensor],
        weights: T.Iterable[float],
        *,
        out: T.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Aggregate gradients.

    :param updates: The gradient vectors.
    :param weights: The weights of the updates.
    :param out: Aggregate the updates to this tensor.
    :return: The aggregated update.

    :raise ValueError: If the number of gradients and weights are not equal.
    """
    if out is None:
        out = 0  # type: ignore
    else:
        out.zero_()

    for weight, update in zip(weights, updates, strict=True):
        out += weight * update

    return out  # type: ignore


@torch.no_grad()
def aggregate_vector(
        global_vector: torch.Tensor,
        local_vectors: T.Iterable[torch.Tensor],
        weights: T.Iterable[float],
        *,
        out: T.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Aggregate vectors.

    :param global_vector: The vector of the global model.
    :param local_vectors: The vectors of the local models.
    :param weights: The weights of the local vectors.
    :param out: Aggregate the vectors to this tensor.
    :return: The aggregated vector.
    """
    updates = (vector - global_vector for vector in local_vectors)
    update = aggregate_update(updates, weights)

    return torch.add(global_vector, update, out=out)


@torch.no_grad()
def aggregate_model(
        global_model: nn.Module, local_models: T.Iterable[nn.Module], weights: T.Iterable[float]
) -> nn.Module:
    """
    Aggregate models.

    :param global_model: The global model.
    :param local_models: The local models
    :param weights: The weights of the local models.
    :return: The aggregated model, which is the global model.
    """
    global_vector = model_to_vector(global_model)
    local_vectors = (model_to_vector(local_model) for local_model in local_models)

    global_vector = aggregate_vector(
        global_vector, local_vectors, weights, out=global_vector
    )
    vector_to_model(global_vector, global_model)

    return global_model


@torch.no_grad()
def layer_map(
        function: T.Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
        models: T.Sequence[nn.Module],
        *,
        out: T.Optional[nn.Module] = None,
) -> nn.Module:
    """
    Map a function to the parameters of the layers of the models.

    :param function: The function to map, which takes a tuple of tensors as input and returns a tensor.
    :param models: The models.
    :param out: The model to store the result.
    :return: The mapped model.
    """
    out = out or copy.deepcopy(models[0])

    for out_param, *params in zip(
            out.parameters(), *[model.parameters() for model in models]
    ):
        params = tuple([param.data for param in params])
        result = function(params)
        out_param.data.copy_(result)

    return out


# From PyTorch
def _check_param_device(param: torch.Tensor, old_param_device: T.Optional[int]) -> int:
    """
    Check if the parameter is located in the same device as the previous parameters.
    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device
