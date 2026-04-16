import torch
from torch import nn

from .flow import Flow
from utils import checkNan

class RNVP(Flow):
    def __init__(self, maskList, tList, sList, prior = None, name = "RNVP"):
        super(RNVP,self).__init__(prior,name)

        assert len(tList) == len(sList)
        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList,requires_grad=False)
        self.maskListR = nn.Parameter(1-maskList,requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)
        self.sList = torch.nn.ModuleList(sList)

    def inverse(self,y):
        inverseLogjac = y.new_zeros(y.shape[0])
        for i in range(len(self.tList)):
            y_ = y*self.maskList[i]
            s = self.sList[i](y_)*self.maskListR[i]
            t = self.tList[i](y_)*self.maskListR[i]
            y = y_ + self.maskListR[i] * (y * checkNan(torch.exp(s)) + t)
            for _ in y.shape[1:]:
                s = s.sum(dim=-1)
            inverseLogjac += s
        return y,inverseLogjac
    
    # def inverse(self, y):
    #     inverseLogjac = y.new_zeros(y.shape[0])
    #     for i in range(len(self.tList)):
    #         # 提取被 Mask 遮住、保持不变的维度
    #         y_ = y * self.maskList[i]
            
    #         # --- Z2 等变性核心改造开始 ---
    #         # 1. 正向输入
    #         net_s_pos = self.sList[i](y_)
    #         net_t_pos = self.tList[i](y_)
            
    #         # 2. 反向输入 (代入 -y_)
    #         net_s_neg = self.sList[i](-y_)
    #         net_t_neg = self.tList[i](-y_)
            
    #         # 3. 强行构造偶函数 s 和奇函数 t
    #         s_even = (net_s_pos + net_s_neg) / 2.0
    #         t_odd  = (net_t_pos - net_t_neg) / 2.0
            
    #         # 4. 乘上反向 Mask，应用到需要变换的维度上
    #         s = s_even * self.maskListR[i]
    #         t = t_odd * self.maskListR[i]
    #         # --- Z2 等变性核心改造结束 ---
            
    #         # 核心的仿射耦合变换，完全保持原样
    #         y = y_ + self.maskListR[i] * (y * checkNan(torch.exp(s)) + t)
            
    #         # 雅可比行列式的计算也完全不需要改
    #         for _ in y.shape[1:]:
    #             s = s.sum(dim=-1)
    #         inverseLogjac += s
            
    #     return y, inverseLogjac

    def forward(self,z):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            z_ = self.maskList[i]*z
            s = self.sList[i](z_)*self.maskListR[i]
            t = self.tList[i](z_)*self.maskListR[i]
            z = self.maskListR[i] * (z - t) * checkNan(torch.exp(-s)) + z_
            for _ in z.shape[1:]:
                s = s.sum(dim=-1)
            forwardLogjac -= s
        return z,forwardLogjac
    
    # def forward(self, z):
    #     forwardLogjac = z.new_zeros(z.shape[0])
    #     # 必须倒序穿过所有层
    #     for i in reversed(range(len(self.tList))):
    #         # 提取保持不变的维度
    #         z_ = self.maskList[i] * z
            
    #         # --- Z2 等变性核心改造开始 (与 inverse 逻辑完全一致) ---
    #         net_s_pos = self.sList[i](z_)
    #         net_t_pos = self.tList[i](z_)
            
    #         net_s_neg = self.sList[i](-z_)
    #         net_t_neg = self.tList[i](-z_)
            
    #         s_even = (net_s_pos + net_s_neg) / 2.0
    #         t_odd  = (net_t_pos - net_t_neg) / 2.0
            
    #         s = s_even * self.maskListR[i]
    #         t = t_odd * self.maskListR[i]
    #         # --- Z2 等变性核心改造结束 ---
            
    #         # 逆向仿射变换 (移项还原 z)
    #         z = self.maskListR[i] * (z - t) * checkNan(torch.exp(-s)) + z_
            
    #         # 雅可比行列式减去相应的缩放项
    #         for _ in z.shape[1:]:
    #             s = s.sum(dim=-1)
    #         forwardLogjac -= s
            
    #     return z, forwardLogjac