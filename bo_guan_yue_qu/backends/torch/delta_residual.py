# 用 PyTorch自身借口实现的 ForwardHook
from torch.nn import Module as StatefulFun
from torch.utils.hooks import RemovableHandle
from .. import YueQuHookingContract


class DeltaResidualHookingContract(YueQuHookingContract):
    def __init__(self, function:StatefulFun, 
                pre_hook:StatefulFun, 
                post_hook:StatefulFun,
                delta_name:str = "约取"):
        self.function = function
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.delta_name = delta_name
        self.handles = []
        
    def attach(self) -> None:
        super().attach()
        pre_hook_handler = self.function.register_forward_pre_hook(self.pre_hook)
        post_hook_handler = self.function.register_forward_hook(self.post_hook)
        self.handles+=[pre_hook_handler, post_hook_handler]
        
    def detach(self) -> None:
        super().detach()
        for handle in self.handles:
            handle.remove()
            self.handles.remove(handle)
        
