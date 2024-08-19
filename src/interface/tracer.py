"""
tracer.py

Provides functionality for tracing a PyTorch model to TorchScript,
for use in C++ LibTorch code.
"""

import os
import time

import torch


def trace_model(
    model_path: str,
    example: torch.Tensor,
    save_path: str,
    model_class: torch.nn.Module,
    model_kwargs: dict
):
    """
    Trace a PyTorch model and save it to a file.

    Args:
        model (nn.Module): the PyTorch model to trace
        save_path (str): the path to save the traced model
    """
    state_dict = torch.load(model_path)
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict)

    traced_model: torch.ScriptModule = torch.jit.trace(model, example)
    traced_model.save(save_path + '.tmp')
    while not os.path.exists(save_path + '.tmp'):
        print('Waiting for trace file to be written...')
        time.sleep(0.1)
    print("Traced model saved to", save_path + '.tmp')
    os.rename(save_path + '.tmp', save_path)
    print("Traced model saved to", save_path)
