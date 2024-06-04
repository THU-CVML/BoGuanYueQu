from dataclasses import dataclass
import torch.nn as nn
@dataclass
class YueQuLayer(nn.Module):
    def __post_init__(self):
        # dataclass生成的init是没有调用super().__init__()的，所以需要手动调用
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.__post_init__
        # 这里调用PyTorch的init，接下来用户写self.xx = xx就能注册参数、子模块之类的。
        super().__init__() 
        # 为了防止用户自己忘记写 super().__post_init__() ，我们换个名字方便用户记忆。
        self.setup()
    def setup(self):
        # 用户实现，初始化增量神经网络的增量参数v
        pass
    def __repr__(self):
        return super().__repr__()
    def extra_repr(self) -> str:
        return super().extra_repr()
    
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        original_repr = cls.__repr__
        original_extra_repr = cls.extra_repr
        dataclass(cls) # 这个3.10以后是in place的， 不保证？
        dataclass_repr = cls.__repr__
        def extra_repr(self):
            dcr = dataclass_repr(self)
            dcr = dcr[dcr.index("(")+1:dcr.rindex(")")]
            return dcr+original_extra_repr(self)
        # cls.extra_repr = lambda self:(dataclass_repr(self)+original_extra_repr(self)) # dataclass的 repr提供给PyTorch
        cls.extra_repr = extra_repr # dataclass的 repr提供给PyTorch
        cls.__repr__ = original_repr
        

if __name__ == "__main__":
    import torch
    from bigmodelvis import Visualization

    # 多写几个dataclass没有毛病，所以就能解决标注问题。
    @dataclass
    @dataclass
    @dataclass
    class TestFlaxStyle(YueQuLayer):
        name:str
        age:int
        def setup(self):
            print(self.name, self.age)
            self.linear = nn.Linear(self.age, self.age)
        def forward(self, x):
            x = torch.Tensor([x])
            return self.linear(x)
        def extra_repr(self) -> str:
            return super().extra_repr()+" Hello World!"
    test = TestFlaxStyle("hello", 1)
    print(test) # str
    print(repr(test))
    print(test(1))
    Visualization(test).structure_graph()
    class TestTorchStyle(YueQuLayer):
        def __init__(self, name, age):
            super().__init__()
            self.name = name
            self.age = age
            print(self.name, self.age)
            self.linear = nn.Linear(self.age, self.age)
        def forward(self, x):
            x = torch.Tensor([x])
            return self.linear(x)
        def extra_repr(self) -> str:
            return super().extra_repr()+" Hello World!"
    test = TestTorchStyle("hello", 1)
    print(test) # str
    print(repr(test))
    print(test(1))
    Visualization(test).structure_graph()
    
          