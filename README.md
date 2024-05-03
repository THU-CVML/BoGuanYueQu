# 博观约取——支持微调视觉模型的增量微调库 (BoGuanYueQu: A Delta Tuning Library that Supports Tuning Computer Vision Models) 



本框架基于开源仓库 https://github.com/thunlp/OpenDelta 开发，
thunlp/OpenDelta 是本框架代码的后端之一，提供了重要的基础功能。

本框架的另一个后端是我们开发的开源框架delta_residual, 该框架目前功能不够稳定，我们目前推荐使用thunlp/OpenDelta作为后端。

## Quick Start
假设你原本的代码流程是这样的
```python
# load the model 
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base")
# doing some awesome things, like finetuning the loaded pretrained model on an awesome dataset.
dataset = AutoDataset()
trainer = AutoTrainer()
trainer.train(model, dataset)
```
现在你想对model进行Finetuning，但是Full Finetuning耗费较多的时空资源。
于是你想通过参数高效微调方法，比如Prefix-Tuning, 进行`博观约取`。
```python
  # load the model 
  from transformers import AutoModelForImageClassification
  model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base")
  # get a yuequ model (or called delta model)
+ from bo_guan_yue_qu import PrefixTuningConfig, PrefixTuningModel
+ delta = PrefixTuningModel(model, 
+                     config=PrefixTuningConfig(
+                         prefix_token_num=25
+                         ), 
+                     modified_modules=['key', 'value'])
+ delta.freeze_module(exclude=["deltas"])
+ delta.log()
  # doing some awesome things, like finetuning the loaded pretrained model on an awesome dataset.
  dataset = AutoDataset()
  trainer = AutoTrainer()
  trainer.train(model, dataset)
```

你还可以对模型的参数效率（Parameter Efficiency）进行检查，这个指标是指相对于全量微调而言我们使用参数高效微调方法进行训练的参数量比值。注意，这个指标和trainable_parameter_ratio不是一回事。
```python
from bo_guan_yue_qu.benchmark import parameter_efficiency, trainable_parameter_ratio
trainable_parameter_ratio(delta), parameter_efficiency(delta)
```
