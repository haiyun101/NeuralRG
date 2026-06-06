import torch
from torch import nn
import numpy as np

import utils
import flow
import train
import source

#from profilehooks import profile
import math
import h5py
import argparse

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='')
parser.add_argument("-folder", default=None)
parser.add_argument("-name", default=None, help='name of flow')

group = parser.add_argument_group('learning  parameters')
group.add_argument("-epochs", type=int, default=1000, help="")
group.add_argument("-batch", type=int, default=32, help="")
group.add_argument("-cuda", type=int, default=-1, help="use GPU")
group.add_argument("-double", action='store_true', help="use float64")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")
group.add_argument("-savePeriod", type=int, default=10, help="")
group.add_argument("-alpha", type=float, default=1, help="")
group.add_argument("-skipHMC", action='store_true', help="")

group = parser.add_argument_group('network parameters')
group.add_argument("-load", action='store_true', help="if load from folder")
group.add_argument("-nlayers", type=int, default=4, help="# of layers in RNVP block")
group.add_argument("-nmlp",type = int, default=2,help="# of layers in MLP")
group.add_argument("-nhidden", type=int, default=32, help="")
group.add_argument("-nrepeat", type=int, default=2, help="repeat of mera block")
group.add_argument("-depthMERA", type=int, default=-1, help="maximum depth of MERA flow")
group.add_argument("-symmetry", action='store_true', help="")
# 【新增】：添加我们的两个物理先验开关
group.add_argument("-weightTying", action='store_true', help="Tie weights across MERA layers for scale invariance")
group.add_argument("-haarPrior", action='store_true', help="Force majority vote using Haar transform")
# add MCMC data drivin search toggle
group.add_argument("-dataDriven", action='store_true', help="Use Data-Driven (Forward KL) training instead of Energy-Based")
group.add_argument("-dataPath", default=None, type=str, help="Path to MCMC data. If None, auto-searches based on L and T")
group.add_argument("-noDeq", action='store_true', help="Disable dequantization noise (use for pre-converted HS continuous samples)")
group.add_argument("-jsLoss", action='store_true', help="Symmetrized-KL (JS-like) training: jsLambda*KL(q||p) + (1-jsLambda)*KL(p||q). Uses HS data + flow samples each step.")
group.add_argument("-jsLambda", type=float, default=0.5, help="Mixing weight for jsLoss: 1.0 = pure reverse-KL, 0.0 = pure forward-KL, 0.5 = symmetric JS-like.")
group.add_argument("-jsMemOpt", action='store_true', help="Memory-optimized JS: do two sequential backward passes (rev then fwd) instead of holding both graphs alive simultaneously. Halves peak GPU memory at ~5-10%% wall-clock cost.")
group.add_argument("-entropyBeta", type=float, default=0.0, help="Forward-KL entropy regularizer: loss += -beta*H(q). Maximizes flow entropy to widen the bridge region. Only active with -dataDriven. ~doubles per-step cost (adds a sampling pass).")
group.add_argument("-flowType", type=str, default="rnvp", choices=["rnvp", "nsf"], help="Coupling layer type: rnvp (affine) or nsf (rational-quadratic spline, more expressive for bridge regions).")
group.add_argument("-nsfBins", type=int, default=8, help="Number of bins K in the NSF rational-quadratic spline (default 8). Only used with -flowType nsf.")
group.add_argument("-nsfBound", type=float, default=5.0, help="Boundary B for the NSF spline: transform is identity outside [-B, B]. Should be > max|data| (e.g. 5*sigma for HS data).")
group.add_argument("-gradClip", type=float, default=0.0, help="Max gradient norm (clip_grad_norm_); 0 disables. Recommended ~5.0 for NSF L=32 where unclipped runs NaN around ep 3-6k.")
group.add_argument("-bridgeWeight", type=float, default=0.0, help="Bridge-targeted upweighting: each training sample with |M|<bridgeThresh gets weight (1+bridgeWeight) instead of 1. 0 disables.")
group.add_argument("-bridgeThresh", type=float, default=0.5, help="Per-sample magnetization threshold for the bridge region (|M_i|<thresh -> upweighted). Only used with -bridgeWeight>0.")
group.add_argument("-pathGrad", action='store_true', help="Reverse-KL path-gradient (STL) estimator: backward through path only, drop explicit-theta score-function term. Vaitl et al. 2024 (arxiv:2403.15881). Adds one inverse pass per step (~50%% wall-time overhead). Reverse-KL only for now.")

group = parser.add_argument_group('Ising target parameters')
#
group.add_argument("-L",type=int, default=4,help="linear size")
group.add_argument("-d",type=int, default=2,help="dimension")
group.add_argument("-T",type=float, default=2.269185314213022, help="Temperature")

args = parser.parse_args()

if args.folder is None:
    rootFolder = './opt/replyMERA_ising_' + str(args.L)+"_T_"+str(args.T)+"_depthLevel_"+str(args.depthMERA)+"_MERA"+'_l'+str(args.nlayers)+'_M'+str(args.nmlp)+'_H'+str(args.nhidden)+'_R'+str(args.nrepeat)+"/"
    print("No specified saving path, using",rootFolder)
else:
    rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'

utils.createWorkSpace(rootFolder)
if args.load:
    with h5py.File(rootFolder+"/parameters.hdf5","r") as f:
        epochs = int(np.array(f["epochs"]))
        batch = int(np.array(f["batch"]))
        cuda = int(np.array(f["cuda"]))
        double = bool(np.array(f["double"]))
        lr = float(np.array(f["lr"]))
        savePeriod = int(np.array(f["savePeriod"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        depthMERA = int(np.array(f["depthMERA"]))
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        T = float(np.array(f["T"]))
        # 【新增读取】
        weightTying = bool(np.array(f["weightTying"])) if "weightTying" in f else False
        haarPrior = bool(np.array(f["haarPrior"])) if "haarPrior" in f else False
        flowType = str(np.array(f["flowType"]).item().decode()) if "flowType" in f else "rnvp"
        nsfBins  = int(np.array(f["nsfBins"]))  if "nsfBins"  in f else 8
        nsfBound = float(np.array(f["nsfBound"])) if "nsfBound" in f else 5.0
else:
    epochs = args.epochs
    batch = args.batch
    cuda = args.cuda
    double = args.double
    lr = args.lr
    savePeriod = args.savePeriod
    nlayers = args.nlayers
    nmlp = args.nmlp
    nhidden = args.nhidden
    nrepeat = args.nrepeat
    depthMERA = args.depthMERA
    L = args.L
    d = args.d
    T = args.T
    # 【新增赋值】
    weightTying = args.weightTying
    haarPrior = args.haarPrior
    flowType = args.flowType
    nsfBins  = args.nsfBins
    nsfBound = args.nsfBound
    
    with h5py.File(rootFolder+"parameters.hdf5","w") as f:
        f.create_dataset("epochs",data=args.epochs)
        f.create_dataset("batch",data=args.batch)
        f.create_dataset("cuda",data=args.cuda)
        f.create_dataset("double",data=args.double)
        f.create_dataset("lr",data=args.lr)
        f.create_dataset("savePeriod",data=args.savePeriod)
        f.create_dataset("nlayers",data=args.nlayers)
        f.create_dataset("nmlp",data=args.nmlp)
        f.create_dataset("nhidden",data=args.nhidden)
        f.create_dataset("nrepeat",data=args.nrepeat)
        f.create_dataset("depthMERA",data=args.depthMERA)
        f.create_dataset("L",data=args.L)
        f.create_dataset("d",data=args.d)
        f.create_dataset("T",data=args.T)
        # 【新增保存】
        f.create_dataset("weightTying",data=weightTying)
        f.create_dataset("haarPrior",data=haarPrior)
        f.create_dataset("flowType", data=np.string_(flowType))
        f.create_dataset("nsfBins",  data=nsfBins)
        f.create_dataset("nsfBound", data=nsfBound)
        f.create_dataset("bridgeWeight", data=args.bridgeWeight)
        f.create_dataset("bridgeThresh", data=args.bridgeThresh)

device = torch.device("cpu" if cuda<0 else "cuda:"+str(cuda))

if double:
    dtype = torch.float64
else:
    dtype = torch.float32

target = source.Ising(L, d, T)
target = target.to(device=device,dtype=dtype)

if args.name is None:
    name = "SymmMERA"+'_l'+str(nlayers)+'_M'+str(nmlp)+'H'+str(nhidden)+'_R'+str(nrepeat)+'_Ising'
else:
    name = args.name

if args.symmetry:
    def op(x):
        return -x

    sym = [op]
    print("Using symmetry")
else:
    sym = None

if depthMERA == -1:
    depthMERA = None
# fw = train.symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,sym,device,dtype,name,depthMERA=depthMERA)
# 【修改为】：
fw = train.symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,sym,device,dtype,name,
                            depthMERA=depthMERA,
                            weightTying=weightTying,
                            haarPrior=haarPrior,
                            flowType=flowType, nsfBins=nsfBins, nsfBound=nsfBound)

#fw = train.symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,sym,device,dtype,name)

if args.load:
    import os
    import glob
    name = max(glob.iglob(rootFolder+'savings/*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = torch.load(name)
    fw.load(saved)

def measure(x):
        p = torch.sigmoid(2.*x).reshape(-1, target.nvars[0])
        s = 2.*p.data.cpu().numpy() - 1.
        sf = (s.mean(axis=1))**2 - (s**2).sum(axis=1)/target.nvars[0]**2  +1./target.nvars[0] #structure factor
        return  sf


# --- MODIFY THIS LINE ---
LOSS,ZACC,ZOBS,XACC,XOBS = train.learnInterface(
    target, fw, batch, epochs, save=True, saveSteps=savePeriod,
    savePath=rootFolder, measureFn=measure, alpha=args.alpha,
    skipHMC=args.skipHMC, dataDriven=args.dataDriven,
    dataPath=args.dataPath, targetT=args.T, noDeq=args.noDeq,
    jsLoss=args.jsLoss, jsLambda=args.jsLambda, jsMemOpt=args.jsMemOpt,
    entropyBeta=args.entropyBeta, gradClip=args.gradClip,
    bridgeWeight=args.bridgeWeight, bridgeThresh=args.bridgeThresh,
    pathGrad=args.pathGrad,
)
#LOSS,ZACC,ZOBS,XACC,XOBS = train.learnInterface(target,fw,batch,epochs,save=True,saveSteps = savePeriod,savePath=rootFolder,measureFn = measure)
