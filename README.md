# Durability-Optimized Model Poisoning Attack against Federated Learning Systems

This is code implementation for `Durability-Optimized Model Poisoning Attack against Federated Learning Systems`, check code for details.

## Project Structure

```bash
./
├─ attackers  # Code for MPAs
├─ flcore  # A simple and lightweight federated learning framework
└─ system  # Experiment environment, we implemented FedAvg here
```

## How to Start

We implement CLI in `system/__main__.py`, type `python -m system --help` to see available commands. Note that `optimizer_factory()`  stay in `system/__main__.py` , you need import it first before deserialize.

A comprehensive guide is coming soon.

