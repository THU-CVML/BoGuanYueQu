from typing import Optional, List, Union, Type, ClassVar
from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
from pydantic import BaseModel, ConfigDict, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opendelta.utils.cuda import get_device
from bo_guan_yue_qu.utils import get_module_device, get_tuple_device, auto_tuple_output_for_forward_hook, auto_tuple_output_for_forward_pre_hook
from loguru import logger

class YueQuLayerConfig(BaseModel):
    """参数高效微调算法针对被修改的原子网络的修改方式的配置类。不包含应用参数高效微调的位置的配置。
    """
    algorithm_name:str = "约取"
    supported_type:Type[nn.Module] = nn.Module # 支持微调的类别, 要求继承自nn.Module
    
class YueQuLayer(nn.Module):
    # @auto_tuple_output_for_forward_pre_hook
    def _forward_pre_hook(self, module: nn.Module, inputs: tuple, **kwargs) -> tuple:
        # logger.warning("Shall be implemented by subclasses. ")
        logger.debug(
            f"delta:{self.device} is called in addition to model:{get_module_device(module)} with input:{get_tuple_device(inputs)}."
        )
        return inputs

    # @auto_tuple_output_for_forward_hook # 这个细节用户不应该关心，我们隐藏起来
    def _forward_hook(
        self, module: nn.Module, inputs: tuple, outputs: tuple, **kwargs
    ) -> tuple:
        logger.debug(
            f"delta:{self.device} is called in addition to model:{get_module_device(module)} with input:{get_tuple_device(inputs)}."
        )
        # logger.warning("Shall be implemented by subclasses. ")
        return outputs

class OpenDeltaWrapper4YueQuLayer(nn.Module):
    # 参考  opendelta.utils.data_parallel.sequential_caller
    def __init__(self, yuequ_layer:YueQuLayer, reference_layer: nn.Module=None):
        self.yuequ_layer = yuequ_layer
        self.yuequ_layer._forward_pre_hook = auto_tuple_output_for_forward_pre_hook(self.yuequ_layer._forward_pre_hook)
        self.yuequ_layer._forward_hook = auto_tuple_output_for_forward_hook(self.yuequ_layer._forward_hook)

        # OpenDelta 不支持 传入 orginial module，所以不能动态的读取和策略
        # OpenDelta 认为 修改 ret 和 ret是怎么得到的无关，所以input也只好传None
        # 我们强行传入，使用cache获得上一次的输入，这样就完整了
        self.cache = None
        self.reference_model = reference_layer
        
    def pre_forward(self, *args, **kwargs):

        self.cache = (args, kwargs)
        return self.yuequ_layer._forward_pre_hook(module=self.reference_layer, input=args, **kwargs)
    
    def post_forward(self, ret):
        if self.cache:
            return self.yuequ_layer._forward_hook(module=self.reference_layer, input=self.cache[0], 
                                                  output=ret, **self.cache[1])
        return self.yuequ_layer._forward_hook(module=self.reference_layer, input=None, output=ret)

class YueQuModel(DeltaBase):
    def __init__(self,
                 reference_model: nn.Module,
                 layer_config:YueQuLayerConfig,
                 layer_delta_class:Type[YueQuLayer]= YueQuLayer,
                algorithm_name=None,
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 ):
        DeltaBase.__init__(self,
                           reference_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        self.algorithm_name = algorithm_name or layer_config.algorithm_name
        self.layer_config = layer_config
        self.layer_delta_class = layer_delta_class
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
        yuequ_layer = self.new_module_like(child_module=child_ref)
        delta_layer = OpenDeltaWrapper4YueQuLayer(yuequ_layer, child_ref) # 符合OpenDelta的caller方式
        self.insert_sequential_module(child_ref, 
                                    delta_module=delta_layer, 
                                    delta_name=self.algorithm_name)


    def new_module_like(self, child_module):
        if not isinstance(child_module, self.layer_config.supported_type):
            logger.warning(f"Trying to apply delta method '{self.layer_config.algorithm_name}' to backbone child module with unsupported type '{type(child_module)}'. Behavior may not be well defined. ")
        new_module = self.layer_delta_class(
                reference_layer=child_module,
                layer_config=self.layer_config) 
        # That's it! 具体怎么从child_module读取weight、in_features、out_features等信息，
        # 以及在哪个device上面初始化参数，由子类自己实现。    
        self.delta_modules.append(new_module)
        return new_module
