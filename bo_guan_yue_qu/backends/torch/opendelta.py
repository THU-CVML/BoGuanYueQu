from torch.nn import Module as StatefulFun
from opendelta.utils.data_parallel import sequential_caller, decorate
from opendelta.utils.data_parallel import new_replicate_for_data_parallel as opendelta_data_parallel

from .. import YueQuHookingContract

class OpenDeltaHookingContract(YueQuHookingContract):
    def __init__(self, function:StatefulFun, 
                pre_hook:StatefulFun, 
                post_hook:StatefulFun,
                delta_name:str = "约取"
                # function, old_forward, new_forward, 
                #  old_replicate_for_data_parallel, new_replicate_for_data_parallel
                 ):
        self.function = function
        self.old_forward = function.forward
        self.new_forward = decorate(function.forward, sequential_caller, 
                            extras=(function, delta_name),
                            kwsyntax=True).__get__(function, type(function))
        self.old_replicate_for_data_parallel = function._replicate_for_data_parallel
        self.new_replicate_for_data_parallel = opendelta_data_parallel.__get__(function, type(function))
        
    def attach(self) -> None:
        super().attach()
        self.function.forward = self.new_forward
        self.function._replicate_for_data_parallel = self.new_replicate_for_data_parallel

        
        
    def detach(self) -> None:
        super().detach()
        self.function.forward = self.old_forward
        self.function._replicate_for_data_parallel = self.old_replicate_for_data_parallel
        



    