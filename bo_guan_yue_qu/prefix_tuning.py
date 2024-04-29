from typing import Optional, Union

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
import torch.nn as nn
from opendelta import BaseDeltaConfig
import math
from dataclasses import dataclass, field

from opendelta.delta_models.prefix import PrefixConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PrefixAppendedLinear(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        weight,
        config:PrefixConfig,
        device, 
    ):
        super().__init__()
        self.prefix_token_num = config.prefix_token_num
        # self.num_heads = num_heads
        self.device = device
        self.hidden_dim = out_features
        self.prefix = nn.Parameter(torch.randn(self.prefix_token_num, 
                                                 self.hidden_dim, 
                                                 device=self.device)
                                   , requires_grad=True)
    # 注意参考 opendelta  的 caller，会把 post_forward 函数搞进去。
    # 不是你这api写的好烂，sequential caller用的和Parallel caller还用不一样的东西，有意思吗？
    def post_forward(self, key_or_value:torch.Tensor): 
         # 比如说这是 key = x[b, s, h]@W_k[h, h] 生成的
        #  我们可以这样操作，前提是，Attention类里面key和value是分开写的。
        batch_size, seq_len, hidden_dim = key_or_value.shape
        # if not self.instantiated:
        #     self.hidden_dim = hidden_dim
        #     self.instantiate()
        # 我们想让 batch_size, seq_len, hidden_dim -> batch_size, seq_len+prefix_token_num, hidden_dim
        prefix_appended_kv = torch.cat([self.prefix.expand(batch_size,
                                               self.prefix_token_num, 
                                               self.hidden_dim)
                            ,key_or_value 
                            ], dim=1)
        return prefix_appended_kv




class PrefixTuningModel(DeltaBase):
    r""" 
    这是我实现Prefix的方式; 简洁准确
    """

    config_class = PrefixConfig
    delta_type = "prefix_tuning"
    default_modified_modules = ['attn@.k@', 'attn@.v@'] # prefix方法修改的 common structure
    _need_pseudo_data = False
    def __init__(self,
                 backbone_model: nn.Module,
                 config:PrefixConfig,
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )


    def update_module(self, module: nn.Module, key: str):
        parent_ref, child_name, child_ref = self.find_module(module, key)
        parallel_module = self.new_module_like(child_module=child_ref)
        self.insert_sequential_module(child_ref, 
                                    delta_module=parallel_module, 
                                    delta_name=PrefixTuningModel.delta_type)

    def _pseudo_data_to_instantiate(self, module):
        # no need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module):
        in_features, out_features = child_module.in_features, child_module.out_features
        from opendelta.utils.cuda import get_device
        module_device = get_device(child_module)
        new_module = PrefixAppendedLinear(in_features = in_features,
                                    out_features = out_features,
                                    weight = child_module.weight,
                                    config=self.config, 
                                    device = module_device)        
        self.delta_modules.append(new_module)
        return new_module
