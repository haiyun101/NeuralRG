import torch
from torch import nn
import h5py
import numpy as np
import subprocess
import utils
from utils import HMCwithAccept
from .symmetry import Symmetrized
from torchvision.utils import make_grid, save_image

import flow
import source
import math
from flow import Flow

class HaarRNVP(flow.Flow):
    def __init__(self, rnvp_block, prior=None, name="HaarRNVP"):
        super(HaarRNVP, self).__init__(prior, name)
        self.rnvp = rnvp_block
        # 4x4 正交 Haar 矩阵
        matrix = 0.5 * torch.tensor([
            [ 1.0,  1.0,  1.0,  1.0],  # 通道0：多数表决均值
            [ 1.0,  1.0, -1.0, -1.0],  # 通道1：涨落
            [ 1.0, -1.0,  1.0, -1.0],  # 通道2：涨落
            [ 1.0, -1.0, -1.0,  1.0]   # 通道3：涨落
        ], dtype=torch.float32)
        self.register_buffer('haar_matrix', matrix)

    def forward(self, x):
        # x shape在MERA中被reshape为: (Batch, 1, 2, 2)
        B = x.shape[0]
        x_flat = x.reshape(B, 4)
        # 1. 强制物理先验：正交矩阵相乘
        z_flat = torch.matmul(x_flat, self.haar_matrix.t().to(x.device))
        z = z_flat.reshape(B, 1, 2, 2)
        # 2. 将分离后的慢模和快模送入网络去清除噪声
        z_out, logp = self.rnvp.forward(z)
        return z_out, logp

    def inverse(self, z):
        # 生成过程 (Latent -> Physical)
        y, logp = self.rnvp.inverse(z)
        B = y.shape[0]
        y_flat = y.reshape(B, 4)
        # 逆变换 (正交矩阵的逆即为转置)
        x_flat = torch.matmul(y_flat, self.haar_matrix.to(y.device))
        x = x_flat.reshape(B, 1, 2, 2)
        return x, logp

def symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,symmetryList,device,dtype,name = None, channel = 1, depthMERA = None,  weightTying=False, haarPrior=False):
    s = source.Gaussian([channel]+[L]*d)

    depth = int(math.log(L,2))*nrepeat*2

    coreSize = 4*channel

    MaskList = []
    for _ in range(depth):
        masklist = []
        for n in range(nlayers):
            if n%2 == 0:
                b = torch.zeros(1,coreSize)
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
                b.zero_()[:,i] = 1
                b=b.view(1,channel,2,2)
            else:
                b = 1-b
            masklist.append(b)
        masklist = torch.cat(masklist,0).to(torch.float32)
        MaskList.append(masklist)

    dimList = [coreSize]
    for _ in range(nmlp):
        dimList.append(nhidden)
    dimList.append(coreSize)

    # layers = [flow.RNVP(MaskList[n], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(coreSize)]) for _ in range(nlayers)]) for n in range(depth)]
    layers = []
    
    if weightTying:
        print(">>> Using Physical Prior: Weight Tying (Scale Invariance)")
        # 实例化一次
        shared_mask = MaskList[0]
        shared_tList = [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)]
        shared_sList = [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(coreSize)]) for _ in range(nlayers)]
        rnvp_block = flow.RNVP(shared_mask, shared_tList, shared_sList)
        
        if haarPrior:
            print(">>> Using Physical Prior: Haar Wavelet (Majority Vote)")
            rnvp_block = HaarRNVP(rnvp_block)
            
        # 所有层复用同一个模块
        layers = [rnvp_block for n in range(depth)]
        
    else:
        # 原作者的 Baseline 逻辑：每一层都有独立的参数
        for n in range(depth):
            tList = [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)]
            sList = [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(coreSize)]) for _ in range(nlayers)]
            rnvp_block = flow.RNVP(MaskList[n], tList, sList)
            
            if haarPrior:
                if n == 0: # 只打印一次
                    print(">>> Using Physical Prior: Haar Wavelet (Majority Vote)")
                rnvp_block = HaarRNVP(rnvp_block)
                
            layers.append(rnvp_block)
            
    f = flow.MERA(2,L,layers,nrepeat,depth = depthMERA,prior = s)
    if symmetryList is not None:
        f = Symmetrized(f,symmetryList,name = name)
    f.to(device = device,dtype = dtype)
    return f

# useless
def learn(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None, weight_decay = 0.001, adaptivelr = False, measureFn = None):
    if savePath is None:
        savePath = "./opt/tmp/"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ACC = []
    OBS = []


    for epoch in range(epochs):
        x ,sampleLogProbability = flow.sample(batchSize)
        # # 修改为以下逻辑：
        # halfBatch = batchSize // 2
        # # 1. 采样一半的正向噪声 z
        # z_pos = flow.prior.sample(halfBatch)
        # # 2. 构造对应的负向噪声 -z
        # z_neg = -z_pos
        # # 3. 拼接成完全对称的噪声 batch [z, -z]
        # z_sym = torch.cat([z_pos, z_neg], dim=0)
        
        # # 4. 手动通过网络（代替 flow.sample）
        # # 注意：这样可以保证噪声在输入前是绝对镜像的
        # sampleLogp_z = flow.prior.logProbability(z_sym)
        # x, inverseLogjac = flow.inverse(z_sym)
        # sampleLogProbability = sampleLogp_z - inverseLogjac
        
        #loss = sampleLogProbability.mean() - source.logProbability(x).mean()
        lossorigin = (+sampleLogProbability + source.logProbability(x))
        loss = lossorigin.mean()
        lossstd = lossorigin.std()
        del lossorigin
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch, "L:",loss.item(),"+/-",lossstd.item())

        LOSS.append([loss.item(),lossstd.item()])
        if adaptivelr:
            scheduler.step()
        if save and epoch%saveSteps == 0:
            d = flow.save()
            torch.save(d,savePath+flow.name+".saving")

    return LOSS,ACC,OBS


def learnInterface(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None,keepSavings = 3, weight_decay = 0.001, adaptivelr = False, HMCsteps = 10, HMCthermal = 10, HMCepsilon = 0.2, measureFn = None, alpha=1.0,skipHMC = True):

    def cleanSaving(epoch):
        if epoch >= keepSavings*saveSteps:
            cmd =["rm","-rf",savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch-keepSavings*saveSteps)+".saving"]
            subprocess.check_call(cmd)
            cmd =["rm","-rf",savePath+"records/"+flow.name+"Record_epoch"+str(epoch-keepSavings*saveSteps)+".hdf5"]
            subprocess.check_call(cmd)

    def latentU(z):
        x,_ = flow.inverse(z)
        return -(flow.prior.logProbability(z)+source.logProbability(x)-flow.logProbability(x))

    if savePath is None:
        savePath = "./opt/tmp/"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ZACC = []
    XACC = []
    ZOBS = []
    XOBS = []
    ENERGY = []   # <--- 新增：记录能量
    ENTROPY = []  # <--- 新增：记录微分熵

    z_ = flow.prior.sample(batchSize)
    x_ = flow.prior.sample(batchSize)

    L = int(x_.shape[-1]**0.5)

    for epoch in range(epochs):
        x,sampleLogProbability = flow.sample(batchSize)
 # --- 新增：分离并提取单轮的 Energy 和 Entropy 标量值 ---
        energy_val = -source.logProbability(x).mean().item()
        entropy_val = -sampleLogProbability.mean().item()

        lossorigin = (sampleLogProbability - source.logProbability(x))
        lossstd = lossorigin.std()
        # loss = lossorigin.mean()
        # 原来代码中的强制对称惩罚项
        loss = (lossorigin.mean()+alpha*(sampleLogProbability.mean()-flow.logProbability(-x).mean()))
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        if adaptivelr:
            scheduler.step()

        del sampleLogProbability

        print("epoch:",epoch, "L:",loss.item(),"F:",lossorigin.mean().item(),"+/-",lossstd.item())
        del lossorigin

        LOSS.append([loss.item(),lossstd.item()])
        ENERGY.append(energy_val)   # <--- 新增
        ENTROPY.append(entropy_val) # <--- 新增      

        # if (epoch%saveSteps == 0 and epoch > 50) or epoch == epochs:
        # 将 epoch > 50 改为 epoch > 0，这样如果 saveSteps=10，就会从 10 开始存
        if (epoch > 0 and epoch % saveSteps == 0) or epoch == epochs:
            configuration = torch.sigmoid(2.*x[:100])
            save_image(configuration, savePath+'/proposals_{:04d}.png'.format(epoch), nrow=10, padding=1)
            if skipHMC:
                print("Skip HMC")
                ZACC.append(np.nan)
                XACC.append(np.nan)
                ZOBS.append([np.nan,np.nan])
                XOBS.append([np.nan,np.nan])

            else:
                z_,zaccept = HMCwithAccept(latentU,z_.detach(),HMCthermal,HMCsteps,HMCepsilon)
                x_,xaccept = HMCwithAccept(source.energy,x_.detach(),HMCthermal,HMCsteps,HMCepsilon)
                with torch.no_grad():
                    x_z,_ = flow.inverse(z_)
                    z_last,_ = flow.forward(x_z)

                with torch.no_grad():
                    Zobs = measureFn(x_z)
                    Xobs = measureFn(x_)
                print("accratio_z:",zaccept.mean().item(),"obs_z:",Zobs.mean(),  ' +/- ' , Zobs.std()/np.sqrt(1.*batchSize))
                print("accratio_x:",xaccept.mean().item(),"obs_x:",Xobs.mean(),  ' +/- ' , Xobs.std()/np.sqrt(1.*batchSize))
                ZACC.append(zaccept.mean().cpu().item())
                XACC.append(xaccept.mean().cpu().item())
                ZOBS.append([Zobs.mean().item(),Zobs.std().item()/np.sqrt(1.*batchSize)])
                XOBS.append([Xobs.mean().item(),Xobs.std().item()/np.sqrt(1.*batchSize)])

            if save:
                with torch.no_grad():
                    samples,_ = flow.sample(batchSize)
                with h5py.File(savePath+"records/"+flow.name+"HMCresult_epoch"+str(epoch)+".hdf5","w") as f:
                    if skipHMC:
                        tmpShape = samples.detach().cpu().numpy().shape
                        placeHolder = np.empty(tmpShape)
                        placeHolder[:] = np.nan
                        f.create_dataset("XZ",data=placeHolder)
                        f.create_dataset("Y",data=placeHolder)
                    else:
                        f.create_dataset("XZ",data=x_z.detach().cpu().numpy())
                        f.create_dataset("Y",data=x_.detach().cpu().numpy())
                    f.create_dataset("X",data=samples.detach().cpu().numpy())

                if not skipHMC:
                    del x_z
                del samples

                with h5py.File(savePath+"records/"+flow.name+"Record_epoch"+str(epoch)+".hdf5", "w") as f:
                    f.create_dataset("LOSS",data=np.array(LOSS)[:,0])
                    f.create_dataset("LOSSSTD",data=np.array(LOSS)[:,1])
                    f.create_dataset("ENERGY",data=np.array(ENERGY))   # <--- 新增
                    f.create_dataset("ENTROPY",data=np.array(ENTROPY)) # <--- 新增
                    f.create_dataset("ZACC",data=np.array(ZACC))
                    f.create_dataset("ZOBS",data=np.array(ZOBS))
                    f.create_dataset("XACC",data=np.array(XACC))
                    f.create_dataset("XOBS",data=np.array(XOBS))
                d = flow.save()
                torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
                # cleanSaving(epoch)

        del x

    return LOSS,ZACC,ZOBS,XACC,XOBS
