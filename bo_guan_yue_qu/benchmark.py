from opendelta import DeltaBase
from opendelta.utils.inspect import num_total_parameters, inspect_module_statistics, num_trainable_parameters
def parameter_efficiency(delta:DeltaBase):
    # if delta._delta_info['state'] == 'on':
        # do_detach = True # 似乎有些没有 _delta_info
    do_detach = True
    if do_detach:
        delta.detach()
    backbone_model = delta.backbone_model
    numel_w = num_total_parameters(backbone_model)
    numel_v = num_total_parameters(delta)
    # if do_detach:
    #     delta.attach()
    return numel_v/numel_w

def trainable_parameter_ratio(delta:DeltaBase):
    # stat = inspect_module_statistics(delta.backbone_model)
    # return stat['trainable_ratio']*100
    modified_backbone_model = delta.backbone_model
    return num_trainable_parameters(modified_backbone_model)/num_total_parameters(modified_backbone_model)    

import optuna
from pydantic import BaseModel, Field
from typing import Optional, List, Union
import optuna
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from optuna.distributions import BaseDistribution, IntDistribution, FloatDistribution, CategoricalDistribution

def get_param_distribution(field: Field) -> Union[BaseDistribution, List]:
    """
    根据pydantic字段的类型和Field信息返回optuna的参数分布。
    """
    type_ = field.type_
    if hasattr(type_, '__args__'):  # Optional or Union
        type_ = type_.__args__[0]

    if isinstance(type_, int):
        # 对整数类型，如果提供了ge和le，则使用这些值作为分布的范围
        if field.ge is not None and field.le is not None:
            return IntDistribution(field.ge, field.le)
        else:
            return [field.default]

    elif isinstance(type_, float):
        # 对浮点数类型，如果提供了ge和le，则使用这些值作为分布的范围
        if field.ge is not None and field.le is not None:
            return FloatDistribution(field.ge, field.le)
        else:
            return [float(field.default)]

    elif isinstance(type_, bool):
        # 布尔类型只有True和False两种可能
        return [True, False]

    elif field.type_ is Optional:
        # 处理Optional类型，提供None和内层类型的默认值
        inner_default = field.type_.__args__[0](**{'ge': None, 'le': None})  # 获取内层类型的默认值
        return [None, inner_default]

    else:
        raise NotImplementedError(f"Type {field.type_} is not supported yet.")

def suggest_pydantic_config(trial: optuna.Trial, pydantic_model: type[BaseModel]) -> BaseModel:
    """
    根据提供的pydantic模型类型和optuna试验，生成配置建议。
    """
    model_fields = pydantic_model.__fields__
    config_kwargs = {}

    for field_name, field in model_fields.items():
        distribution = get_param_distribution(field)
        if isinstance(distribution, BaseDistribution):
            if isinstance(distribution, IntDistribution):
                config_kwargs[field_name] = trial.suggest_int(field_name, distribution)
            elif isinstance(distribution, FloatDistribution):
                config_kwargs[field_name] = trial.suggest_float(field_name, distribution)
            elif isinstance(distribution, CategoricalDistribution):
                config_kwargs[field_name] = trial.suggest_categorical(field_name, distribution)
        elif isinstance(distribution, list):
            config_kwargs[field_name] = trial.suggest_categorical(field_name, distribution)

    # 使用生成的参数创建pydantic模型实例
    config = pydantic_model(**config_kwargs)
    return config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bo_guan_yue_qu.concepts import YueQuLayerConfig
def largest_model_under_parameter_budget(backbone_model:nn.Module, 
                                         delta_config_type:type[YueQuLayerConfig]):
    def objective(trial: optuna.Trial) -> float:
        suggested_config = suggest_pydantic_config(trial, delta_config_type)
        # print(f"Suggested config: {suggested_config}")

        return 0.5  # 占位符

    study = optuna.create_study(direction='maximize')  # Create a new study.
    study.optimize(objective, n_trials=100)
    return study.best_value, study.best_params


# def suggest_config_within_budget(config):

# import optuna
# # Define an objective function to be minimized.
# def objective(trial):
#     delta = PrefixModel(model,config=PrefixConfig(
#         prefix_token_num=trial.suggest_int('prefix_token_num', 1, 100)
#         ),modified_modules=['key', 'value'])
#     delta.freeze_module(exclude=["deltas", "aggregation"])
#     from opendelta.utils.inspect import inspect_module_statistics
#     stat = inspect_module_statistics(delta.backbone_model)
#     # obj_value = stat['delta_ratio']*100
#     ratio = stat['trainable_ratio']*100
#     # obj_value = abs(ratio-0.5)\
#     obj_value = ratio if ratio < 0.5 else 0
#     delta.detach()
#     return obj_value # An objective value linked with the Trial object.
    

# study = optuna.create_study(direction='maximize')  # Create a new study.
# study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
# study.best_value, study.best_params