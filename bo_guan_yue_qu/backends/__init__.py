
class YueQuHookingContract:
    # hook结束的管理对象
    # 类似于 RemovableHandle
    def attach(self):
        pass
    def detach(self):
        pass
    remove = detach
    hook_into = attach
    
    def summary(self):
        pass
    
# class YueQuHooksMaker:
#     def make(self)->YueQuHookingContract:
#         pass