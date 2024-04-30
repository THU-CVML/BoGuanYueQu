#%%
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
name = "facebook/dinov2-base"
# model = AutoModelForImageClassification.from_pretrained(name)
original_model = AutoModel.from_pretrained(name)
# type(model)
import transformers
transformers.models.dinov2.modeling_dinov2.Dinov2Model
#%%
from copy import deepcopy
model = deepcopy(original_model)
#%%
# 测试我修改opendelta添加的方法
from bo_guan_yue_qu import PrefixTuningConfig, PrefixTuningModel
delta = PrefixTuningModel(model, 
                    config=PrefixTuningConfig(
                        # prefix_token_num=100
                        # prefix_token_num=50
                        prefix_token_num=25
                        ), 
                    modified_modules=['key', 'value'])
delta.freeze_module(exclude=["deltas", "aggregation"])
delta.log()
#%%
from bo_guan_yue_qu.benchmark import parameter_efficiency, trainable_parameter_ratio
trainable_parameter_ratio(delta), parameter_efficiency(delta), trainable_parameter_ratio(delta)
#%%
delta.detach()
delta.log()
#%% 
delta.attach()
delta.log()
#%%
# delta._register_delta_infos(delta, None)
# delta._delta_info['method'] 
# %%
