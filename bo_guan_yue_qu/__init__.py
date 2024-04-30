"""博观约取——支持微调视觉模型的增量微调库 (BoGuanYueQu: A Delta Tuning Library that Supports Tuning Computer Vision Models) """

__version__ = "0.1.0"

from bo_guan_yue_qu.prefix_tuning import PrefixTuningModel, PrefixTuningConfig
VPTModel = PrefixTuningModel
VPTConfig = PrefixTuningConfig