import inspect
import threading
from functools import partial
from typing import Any, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loguru import logger

def get_module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def get_tuple_device(t: Tuple) -> torch.device:
    for item in t:
        if isinstance(item, torch.Tensor):
            return item.device
    return None

def make_tuple_or_tensor_to_tuple(tuple_or_tensor: Tuple | torch.Tensor) -> Tuple[Tuple, bool]:
    not_tuple = False
    if not isinstance(tuple_or_tensor, tuple):
        not_tuple = True
        tuple_or_tensor = (tuple_or_tensor,)
    return tuple_or_tensor, not_tuple
        

def auto_tuple_output_for_forward_hook(
    unwrapped_hook: Callable[[nn.Module, nn.Module, Tuple, Tuple], Tuple], 
    return_as_is = False
) -> Callable[[nn.Module, nn.Module, Tuple | torch.Tensor, Tuple | torch.Tensor], Tuple]:
    def wrapped_hook(
        self:nn.Module, module: nn.Module, inputs: tuple | torch.Tensor, outputs: tuple | torch.Tensor
    ) -> tuple:
        _, inputs = make_tuple_or_tensor_to_tuple(inputs)
        not_tuple, outputs = make_tuple_or_tensor_to_tuple(outputs)
        new_outputs = unwrapped_hook(self, module, inputs, outputs)
        return new_outputs[0] if not_tuple and not return_as_is else new_outputs
    return wrapped_hook

def auto_tuple_output_for_forward_pre_hook(
    unwrapped_hook: Callable[[nn.Module, nn.Module, Tuple], Tuple], 
    return_as_is = False
) -> Callable[[nn.Module, nn.Module, Tuple | torch.Tensor], Tuple]:
    def wrapped_hook(
        self:nn.Module, module: nn.Module, inputs: tuple | torch.Tensor
    ) -> tuple:
        not_tuple, inputs = make_tuple_or_tensor_to_tuple(inputs)
        new_inputs = unwrapped_hook(self, module, inputs)
        return new_inputs[0] if not_tuple and not return_as_is else new_inputs

    return wrapped_hook


def set_requires_grad(model: nn.Module, requires_grad: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad