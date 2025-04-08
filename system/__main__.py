import copy
import functools
import importlib as imp
import inspect
import itertools
import random
from pathlib import Path
from typing import Annotated, Callable

import json5
import typer
from box import Box
from rich.pretty import pprint
from rich.progress import track

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command(name="run", help="Run the simulated federated learning algorithm.")
def run(config: Annotated[Path, typer.Option("-c", "--config", help="The config file.", exists=True)]):
    import torch
    import torch.utils.data as D
    import flcore.utils.atomic_io as atomic_io
    from . import _datasets as datasets

    with open(config, encoding='UTF-8') as f:
        config = json5.load(f)
        config = Box(config, camel_killer_box=True)

    pprint(config.to_dict())

    # prepare
    if (seed := config.get("seed", None)) is not None:
        set_seed(seed)

    client_type, server_type, system_type = load_protocol(config.protocol)
    fl_dataset: datasets.FlDataset = atomic_io.load(config.dataset)

    # create server
    robust_fn = None
    if robust_setting := config.server_settings.robust_settings:
        robust_type = load_from_module("flcore.utils.robust", robust_setting.type)
        robust_fn = proper_call(robust_type, **robust_setting)

    model_type = load_from_module("system._models", config.server_settings.model_settings.type)
    model = proper_call(model_type,
                        **config.server_settings.model_settings | dict(num_class=fl_dataset.others["num_class"]))

    server = proper_call(server_type, model=model, robust_fn=robust_fn, **config.server_settings)

    # create clients
    if "num_class" not in fl_dataset.others:
        raise NotImplementedError("Currently only support classification tasks.")

    clients = []

    optimizer_type = load_from_module("torch.optim", config.client_settings.optimizer_settings.type)
    attacker_ids = (
        random.choices(list(fl_dataset.client_data.keys()), k=config.attacker_settings.number)
        if config.attacker_settings
        else []
    )
    for id_, client_data in fl_dataset.client_data.items():
        client = proper_call(client_type, **config.client_settings | dict(
            id_=id_,
            device=torch.device(config.client_settings.device),
            dataset_size=client_data.dataset_size,
            num_class=fl_dataset.others["num_class"],
            train_dataloader=proper_call(D.DataLoader, dataset=client_data.train_dataset,
                                         **config.client_settings.dataloader_settings),
            eval_dataloader=proper_call(D.DataLoader, dataset=client_data.eval_dataset,
                                        **config.client_settings.dataloader_settings),
            test_dataloader=proper_call(D.DataLoader, dataset=client_data.test_dataset,
                                        **config.client_settings.dataloader_settings),
            optimizer_factory=functools.partial(optimizer_factory, optimizer_type=optimizer_type, config=config),
            loss_fn=load_from_module("torch.nn", config.client_settings.loss_fn)(),
        ))

        # compromise some clients
        if id_ in attacker_ids:
            attacker_type = load_from_module("attackers", config.attacker_settings.type)
            attacker = proper_call(
                attacker_type,
                client=client,
                base_model=copy.deepcopy(server.model),  # MpafAttacker
                **config.attacker_settings | dict(
                    attack_range=range(*config.attacker_settings.get("attack_range", [9999])),
                )
            )
            client = attacker

        clients.append(client)

    # register clients to server
    for client in clients:
        server.register_client(client)
        if config.get("fast_mode", False):
            client.connect()

    # create system and then run
    system = proper_call(system_type, server=server, **config.system_settings)

    with open(system.log_dir / "config.json5", 'w', encoding='UTF-8') as f:
        json5.dump(config.to_dict(), f, indent=2)

    system.run()


@app.command(name="gendata", help="Generate dataset for the Federated Learning.")
def gendata(config: Annotated[Path, typer.Option("-c", "--config", help="The config file", exists=True)]):
    import flcore.utils.atomic_io as atomic_io

    from . import _datasets as datasets

    with open(config, encoding='UTF-8') as f:
        config = json5.load(f)
        config = Box(config, camel_killer_box=True)

    # print config details
    pprint(config.to_dict())

    if (seed := getattr(config, "seed", None)) is not None:
        set_seed(seed)

    # load fl dataset
    dataset_loader = getattr(datasets, config.dataset)
    fl_dataset: datasets.FlDataset = proper_call(dataset_loader, **config.dataset_settings)

    # save
    atomic_io.dump(fl_dataset, config.output)

    typer.secho(f"The FlDataset {config.dataset} is saved to {config.output}.", fg=typer.colors.GREEN)


@app.command(name="plot", help="Plot the data distribution of the FlDataset.")
def plot(
        dataset: Annotated[Path, typer.Option("-d", "--dataset", help="The FlDataset file.", exists=True)],
        sort: Annotated[bool, typer.Option(help="Sort the data distribution by the number of samples.")] = False,
):
    import matplotlib
    import matplotlib.pyplot as plt

    import flcore.utils.atomic_io as atomic_io
    import flcore.utils.data as data_utils

    from . import _datasets as datasets

    fl_dataset: datasets.FlDataset = atomic_io.load(dataset)

    typer.secho("Analyzing, this may take a long while.", fg=typer.colors.GREEN)

    num_class = fl_dataset.others.get("num_class", None)
    if num_class is None:
        raise NotImplementedError

    colors = itertools.cycle(list(matplotlib.colors.TABLEAU_COLORS.keys()))
    class_color = {}

    # plot
    plt.figure(figsize=(10, 10))

    client_data = fl_dataset.client_data
    if sort:
        client_data = dict(sorted(client_data.items(), key=lambda x: x[1].dataset_size))

    # for each client
    max_height = 0
    for i, data_info in track(enumerate(client_data.values()), description="Plotting", total=len(client_data)):
        targets = [
            *data_utils.get_targets(data_info.train_dataset),
            *data_utils.get_targets(data_info.eval_dataset),
            *data_utils.get_targets(data_info.test_dataset),
        ]

        if len(targets) > max_height:
            max_height = len(targets)

        # for each class
        class_targets = data_utils.collect_targets(targets)
        offset = 0
        for class_ in sorted(class_targets.keys()):
            if class_ not in class_color:
                class_color[class_] = next(colors)

            num_targets = len(class_targets[class_])
            plt.fill_between([i, i + 1], offset, offset + num_targets,
                             facecolor=class_color[class_])
            offset += num_targets

    plt.title("Data Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.xlim(0, len(fl_dataset.client_data))
    plt.ylim(0, max_height)

    plt.savefig(dataset.with_suffix(".png"))

    plt.show()

    typer.secho("Done.", fg=typer.colors.GREEN)


def set_seed(seed):
    import torch
    import random
    import numpy

    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def proper_call(call: Callable, **kwargs):
    sig = inspect.signature(call)

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return call(**kwargs)

    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = call(**kwargs)

    return result


def load_from_module(module_name: str, obj_name: str):
    module = imp.import_module(module_name)
    obj = getattr(module, obj_name)
    return obj


def load_protocol(protocol):
    module_name = f"system.{protocol.lower()}"
    client_type = load_from_module(module_name, protocol + "Client")
    server_type = load_from_module(module_name, protocol + "Server")
    system_type = load_from_module(module_name, protocol)

    return client_type, server_type, system_type


def optimizer_factory(self, optimizer_type, config):
    return proper_call(optimizer_type, params=self.model.parameters(),
                       **config.client_settings.optimizer_settings)


if __name__ == '__main__':
    app()
