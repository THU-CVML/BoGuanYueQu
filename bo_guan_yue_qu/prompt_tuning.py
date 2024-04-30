from typing import Optional, Union

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
import torch.nn as nn
from opendelta import BaseDeltaConfig
import math
from dataclasses import dataclass, field

from opendelta.delta_models.soft_prompt import SoftPromptConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SoftPromptLayer(nn.Module):
    r""" 
    这是我实现Prompt的方式; 简洁准确
    """
    # 被修改的应该是？
    default_modified_modules = ['attn@.k@', 'attn@.v@']
    def __init__(self,
        reference_layer: nn.Module,
        config:SoftPromptConfig,
        original_embedding=None,  # 使用原本的embedding去生成token，这样可能满足约束。但是我认为没必要，本来就可逆的。
        # prepended_inputs:list=["hidden_states"], # 用名字指定
        prepended_inputs: list = [0],  # 用位置指定，应该也是合理的。
        removed_outputs: list = [0],  # 如果空就不remove，对于shallow vpt就是这样，只操作layer1。
        # return的1个也被我们当做是tuple。
        #  假设直接选择出来就是tensor
        # tensor_selector = lambda x:x, # 有时候可以 x[0]
        dim_of_tokens=-2,
        dim_of_hidden=-1,
        dim_of_batches=0,
    ):
        super().__init__()
        # peft方法的参数
        self.soft_token_num = config.soft_token_num
        self.hidden_dim = config.hidden_dim
        self.init_range = config.init_range
        self.original_embedding = original_embedding
        # 初始化prompt
        # self.compiled = False
        self.soft_prompts: torch.Tensor = None
        if config.hidden_dim is not None:
            self.prompts = self.instantiate(config.hidden_dim)
        # 操作位置的指定
        self.prepended_inputs = prepended_inputs
        self.removed_outputs = removed_outputs
        # self.tensor_selector = tensor_selector
        self.dim_of_tokens = dim_of_tokens
        self.dim_of_hidden = dim_of_hidden
        self.dim_of_batches = dim_of_batches
        
    def _forward_pre_hook(self, module: nn.Module, inputs: tuple) -> tuple:
        # 返回新的input
        new_input = list(inputs)  # 因为元组不能Assignment
        for i in self.prepended_inputs:
            # selected_tensor = self.tensor_selector(input[i]) # 得到一个指针
            # selected_tensor = torch.cat([selected_tensor, self.prompts], dim=self.dim_of_tokens) # 并没有修改原来的tensor
            # self.tensor_selector(input[i]) = selected_tensor
            selected_tensor: torch.Tensor = inputs[i]
            b = selected_tensor.shape[self.dim_of_batches]
            new_input[i] = torch.cat(
                [
                    selected_tensor,
                    #   torch.tile(self.soft_prompts, dims=selected_tensor.size()[:self.dim_of_tokens])
                    self.soft_prompts.repeat(b, 1, 1),
                ],
                dim=self.dim_of_tokens,
            )
        return tuple(new_input)

    def _forward_hook(self, module: nn.Module, inputs: tuple, outputs: tuple) -> tuple:
        # 返回新的output
        new_outputs = list(outputs)
        for i in self.removed_outputs:
            # 我想要对 output[i]这个tensor进行操作,
            selected_tensor: torch.Tensor = outputs[i]
            # 我要在 self.dim_of_tokens 这一个维度上，去掉最后添加的 self.soft_token_num 个元素
            new_outputs[i] = selected_tensor.narrow(
                self.dim_of_tokens,
                0,
                selected_tensor.size(self.dim_of_tokens) - self.soft_token_num,
            )
        return tuple(new_outputs)

    def instantiate(self, hidden_dim) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        soft_prompts = torch.FloatTensor(1, self.soft_token_num, hidden_dim)
        if self.original_embedding is not None:
            soft_prompts.data = torch.clone(
                self.original_embedding(
                    torch.tensor([i for i in range(self.soft_token_num)])
                )
            )
        else:
            soft_prompts = soft_prompts.uniform_(-self.init_range, self.init_range)

        self.soft_prompts: torch.Tensor = nn.Parameter(soft_prompts, requires_grad=True)
        # .to(self.device)

    def merge_into(self, layer: nn.Module):
        raise ArithmeticError("General Soft Prompt Tuning cannot be re-parameterized.")


# class SoftPromptModel(YueQuModel):
#     config_class = PrefixConfig
#     delta_type = "prefix_tuning"
#     default_modified_modules = ['attn@.k@', 'attn@.v@'] # prefix方法修改的 common structure
#     _need_pseudo_data = False