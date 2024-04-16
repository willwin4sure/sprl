"""
tracer.py

Provides functionality for tracing a PyTorch model to TorchScript,
for use in C++ LibTorch code.
"""

import torch

def trace_model(model_path: str, example: torch.Tensor, save_path: str):
    """
    Trace a PyTorch model and save it to a file.

    Args:
        model (nn.Module): the PyTorch model to trace
        save_path (str): the path to save the traced model
    """
    model = torch.load(model_path)
    traced_model = torch.jit.trace(model, example)
    traced_model.save(save_path)
