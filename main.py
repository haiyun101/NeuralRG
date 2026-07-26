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
group.add_argument("-volumePreservingWeight", type=float, default=0.0, help="Soft volume-preserving regularizer: loss += lambda * mean((log|det J_MERA|)^2). 0 = off (default). Forces MERA's Jacobian toward 0 so HCG prior variance actually reflects the data marginal variance (rescues σ interpretability). Only active with -dataDriven.")
group.add_argument("-volumePreservingPerLayer", type=int, default=0, help="1 = penalize sum of per-block (log|det J_block|)^2 instead of the total's square. Blocks cannot cancel each other out via expand/contract pairs. Aimed at nr=2 arms that collapse under the standard total VP. Requires -volumePreservingWeight > 0.")
group.add_argument("-pathGrad", action='store_true', help="Reverse-KL path-gradient (STL) estimator: backward through path only, drop explicit-theta score-function term. Vaitl et al. 2024 (arxiv:2403.15881). Adds one inverse pass per step (~50%% wall-time overhead). Reverse-KL only for now.")
group.add_argument("-scaleLoss", type=float, default=0.0, help="Multi-scale loss coefficient lambda_scale. Adds lambda_scale * sum_s MSE(zscore(y_s[::2,::2]), zscore(y_{s+1})) to the training loss. Penalizes deviation from scale invariance at every MERA scale; targets the rev-KL deep-block collapse and fwd-KL deep-block inflation diagnosed in rg_fixed_point_report.md. 0 disables. (forward-KL / dataDriven only for now.)")
group.add_argument("-cosineAnneal", action='store_true', help="Use CosineAnnealingLR scheduler over full training: lr smoothly decays from -lr to cosineEtaMin (default lr*0.01) over -epochs. Off by default; opt-in only (existing runs unaffected). Mutually exclusive in effect with the legacy -adaptivelr StepLR (adaptivelr takes priority if both set).")
group.add_argument("-cosineEtaMin", type=float, default=None, help="Floor LR for cosine schedule. Defaults to lr*0.01 (e.g. 5e-4 -> 5e-6). Only used with -cosineAnneal.")
group.add_argument("-gradAccum", type=int, default=1, help="Gradient accumulation steps: split -batch into K micro-batches of size (batch/K) and accumulate gradients before optimizer.step(). Default 1 = no accumulation (existing behavior). K>=2 lets megabignet + nrepeat>=2 runs preserve effective batch when single-pass batch hits OOM. Only the dataDriven branch implements it; other modes ignore K>1 with a warning. Incompatible with -entropyBeta > 0.")
group.add_argument("-priorType", type=str, default="gaussian", choices=["gaussian", "conditional_gaussian", "hierarchical_conditional_gaussian", "studentT"], help="Latent prior. 'gaussian' is the isotropic N(0,I) baseline. 'conditional_gaussian' = scheme A of improvements_zh.md: single CNN conditions fast on slow. 'hierarchical_conditional_gaussian' = Path C of prior_offload_analysis_zh.md: multi-scale extension of scheme A, one CNN per level (or scale-shared) over strides [L/2, L/4, ..., 1]. 'studentT' = scheme I.1 (heavy-tailed diagonal prior).")
group.add_argument("-priorDf", type=float, default=4.0, help="Degrees-of-freedom for -priorType studentT. Must be > 2 for finite variance. df=4 -> heavy tails with kurtosis at the boundary; df=5+ -> finite excess kurtosis.")
group.add_argument("-condPriorSlowStride", type=int, default=-1, help="slow-grid stride for -priorType conditional_gaussian. Default -1 = max(2, L//4) (4x4 slow for L=16, 8x8 for L=32 etc.). Must divide L.")
group.add_argument("-condPriorHidden", type=int, default=32, help="hidden channels of the conditional-prior CNN (-priorType conditional_gaussian).")
group.add_argument("-hcgScaleShared", type=int, default=1, help="1 = scale-shared single CNN across all HCG levels (RG-invariant conditional whitening, default). 0 = independent CNN per level.")
group.add_argument("-hcgHidden", type=int, default=32, help="hidden channels of the hierarchical-prior CNN(s).")
group.add_argument("-hcgDilated", type=int, default=1, help="1 = per-level CNN uses dilation = coarser-level stride (reaches coarser context). 0 = dilation always 1. Only meaningful with -hcgScaleShared=0.")
group.add_argument("-hcgCircular", type=int, default=1, help="1 = HCG CNN uses padding_mode='circular' (respects Ising periodic BC, default). 0 = zero-padding (matches i2 original but structurally biased at boundaries).")
group.add_argument("-hcgSharedDilations", type=str, default="", help="Progressive dilation for the shared HCG CNN (only meaningful with -hcgScaleShared=1). Comma-separated list of 3 dilations for the CNN's three conv layers, e.g. '1,2,4' → effective RF = 15 (vs default 7 with uniform dilation=1). Empty (default) → uniform dilation=1, RF=7. Useful for L>=32 where Level 1's coarser context is >3 sites away.")
group.add_argument("-hcgInitFromShared", type=str, default="", help="Path to a shared HCG run folder (containing savings/*.saving). Only meaningful with -hcgScaleShared=0 (per-scale). At start of training, loads that shared checkpoint and copies its single CNN's weights into every per-scale CNN — per-scale starts from a scale-invariant point and may then differentiate or stay similar. Useful for testing whether per-scale physics genuinely differs from scale-invariance, given a good starting minimum.")
group.add_argument("-loadFromSmallerL", type=str, default="", help="Path to a smaller-L champion checkpoint (a *.saving file). At start of training, transfers MERA blocks 0..min(N_src,N_tgt)-1 (scale-index aligned) and HCG per-scale CNNs (stride-aligned) from that checkpoint into this run. The larger-L run's extra coarsest MERA blocks + coarsest HCG CNN(s) stay at fresh init. RG-universality warm-start experiment: does a well-trained smaller-L champion give a bigger-L run a head start? Only meaningful when target uses -priorType hierarchical_conditional_gaussian and -hcgScaleShared=0.")
group.add_argument("-loadFromSmallerLStrides", type=str, default="", help="Comma-separated HCG strides (coarsest→finest) of the source ckpt used with -loadFromSmallerL, e.g. '16,8,4,2,1' for an L=32 source. Required whenever -loadFromSmallerL is set. Ensures the per-scale CNN alignment maps by physical stride, not by level index.")
group.add_argument("-loadFromSmallerLComponents", type=str, default="both", choices=["both", "mera", "cnn"], help="Which weights to transfer from the smaller-L source. 'both' (default) = transfer MERA + HCG CNN. 'mera' = only MERA blocks (CNN stays fresh init). 'cnn' = only HCG CNNs (MERA stays fresh init). Ablation to isolate which component contains more transferable physics.")
group.add_argument("-physRegWeightChi", type=float, default=0.0, help="Physical-observable regularizer coefficient λ_χ for susceptibility. Adds λ_χ · (χ(flow_samples) − χ_target)² to the training loss. χ = N·(⟨M²⟩−⟨|M|⟩²) with M = mean sigmoid(2x) per sample (matches main.py:measure() convention). Only fires when >0 AND -dataDriven. Extra cost: one flow.sample of size -physRegBatch per step.")
group.add_argument("-physRegWeightU4", type=float, default=0.0, help="Physical-observable regularizer coefficient λ_U for Binder cumulant U₄=1−⟨M⁴⟩/(3⟨M²⟩²). Same sampling as -physRegWeightChi. Combining χ and U₄ regularizers gives multi-observable physics constraint.")
group.add_argument("-physRegBatch", type=int, default=128, help="Flow-sample batch size for computing χ and U₄ per training step. Larger = less noise but higher cost. 128 samples give ~7% std on χ estimate for L=32.")
group.add_argument("-physRegTargetChi", type=float, default=float("nan"), help="Target χ value (Ising susceptibility at T_c). If NaN (default), auto-compute from the loaded HS dataset once at startup. Override to use exact reference values.")
group.add_argument("-physRegTargetU4", type=float, default=float("nan"), help="Target U₄ (Binder cumulant, ≈0.611 at 2D Ising T_c). If NaN, auto-compute from HS dataset at startup.")
group.add_argument("-bf16", action="store_true", help="Enable bf16 mixed-precision training via torch.cuda.amp.autocast(dtype=bfloat16). A100 Tensor Cores give ~2× speedup on bf16 vs fp32 for conv/linear ops. bf16 has same 8-bit exponent as fp32 so no GradScaler needed. Trade-off: 7-bit mantissa (vs fp32's 23) may destabilize log|det J| accumulator — verify loss trajectory before committing long runs. Only affects the flow's forward passes; optimizer state and gradients accumulate in fp32.")

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
        # Architecture-relevant prior flags must be reloaded for -load to
        # reconstruct the same network; CLI -priorType etc are ignored
        # when -load is set.
        _loaded_priorType = str(np.array(f["priorType"]).item().decode()) if "priorType" in f else "gaussian"
        _loaded_condPriorSlowStride = int(np.array(f["condPriorSlowStride"])) if "condPriorSlowStride" in f else -1
        _loaded_condPriorHidden = int(np.array(f["condPriorHidden"])) if "condPriorHidden" in f else 32
        _loaded_priorDf = float(np.array(f["priorDf"])) if "priorDf" in f else 4.0
        args.priorType = _loaded_priorType
        args.condPriorSlowStride = _loaded_condPriorSlowStride
        args.condPriorHidden = _loaded_condPriorHidden
        args.priorDf = _loaded_priorDf
        # HCG flags (default to sensible values if pre-HCG run)
        args.hcgScaleShared = int(np.array(f["hcgScaleShared"])) if "hcgScaleShared" in f else 1
        args.hcgHidden      = int(np.array(f["hcgHidden"]))      if "hcgHidden"      in f else 32
        args.hcgDilated     = int(np.array(f["hcgDilated"]))     if "hcgDilated"     in f else 1
        args.hcgCircular    = int(np.array(f["hcgCircular"]))    if "hcgCircular"    in f else 1
        args.hcgSharedDilations = str(np.array(f["hcgSharedDilations"]).item().decode()) if "hcgSharedDilations" in f else ""
        # Restore -symmetry too. Historically CLI-only; without this the
        # saved Symmetrized state_dict (has `flow.` prefix) fails to load
        # into a bare MERA. Legacy runs without this field: infer from the
        # checkpoint's key structure below.
        if "symmetry" in f:
            args.symmetry = bool(np.array(f["symmetry"]))
        else:
            # Auto-detect from checkpoint keys (legacy run without the
            # HDF5 flag).
            import glob as _g
            _sav = sorted(_g.glob(rootFolder + 'savings/*.saving'))
            if _sav:
                _tmp = torch.load(_sav[-1], weights_only=False, map_location='cpu')
                _sd = _tmp['model'] if (isinstance(_tmp, dict) and 'model' in _tmp) else _tmp
                _has_flow_prefix = any(k.startswith('flow.') for k in _sd.keys())
                if _has_flow_prefix and not args.symmetry:
                    print("[load] checkpoint has `flow.` prefix → auto-enable -symmetry")
                    args.symmetry = True
                del _tmp, _sd
        # Same restore for other CLI-only flags. If missing from HDF5
        # (legacy run), keep whatever the user passed on CLI.
        if "dataDriven" in f:
            args.dataDriven = bool(np.array(f["dataDriven"]))
        if "noDeq" in f:
            args.noDeq = bool(np.array(f["noDeq"]))
        if "skipHMC" in f:
            args.skipHMC = bool(np.array(f["skipHMC"]))
        if "gradClip" in f:
            args.gradClip = float(np.array(f["gradClip"]))
        if "gradAccum" in f:
            args.gradAccum = int(np.array(f["gradAccum"]))
        if "alpha" in f:
            args.alpha = float(np.array(f["alpha"]))
        if "dataPath" in f:
            _dp = str(np.array(f["dataPath"]).item().decode())
            if _dp:
                args.dataPath = _dp
        if "bf16" in f:
            args.bf16 = bool(np.array(f["bf16"]))
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
        f.create_dataset("volumePreservingWeight", data=args.volumePreservingWeight)
        f.create_dataset("volumePreservingPerLayer", data=args.volumePreservingPerLayer)
        f.create_dataset("scaleLoss", data=args.scaleLoss)
        f.create_dataset("physRegWeightChi", data=args.physRegWeightChi)
        f.create_dataset("physRegWeightU4", data=args.physRegWeightU4)
        f.create_dataset("physRegBatch", data=args.physRegBatch)
        f.create_dataset("physRegTargetChi", data=args.physRegTargetChi)
        f.create_dataset("physRegTargetU4", data=args.physRegTargetU4)
        f.create_dataset("bf16", data=bool(args.bf16))
        f.create_dataset("priorType", data=np.string_(args.priorType))
        f.create_dataset("condPriorSlowStride", data=args.condPriorSlowStride)
        f.create_dataset("condPriorHidden", data=args.condPriorHidden)
        f.create_dataset("priorDf", data=args.priorDf)
        f.create_dataset("hcgScaleShared", data=args.hcgScaleShared)
        f.create_dataset("hcgHidden", data=args.hcgHidden)
        f.create_dataset("hcgDilated", data=args.hcgDilated)
        f.create_dataset("hcgCircular", data=args.hcgCircular)
        f.create_dataset("hcgSharedDilations", data=np.string_(args.hcgSharedDilations))
        # Save -symmetry as a boolean so -load can rebuild the same wrap.
        # Historically -symmetry was CLI-only; state_dicts from Symmetrized
        # runs have `flow.` prefix and won't load into a bare MERA.
        f.create_dataset("symmetry", data=bool(args.symmetry))
        # Same story for other CLI-only flags — save all so `-load` gives
        # exactly the same training loop as the original run.
        f.create_dataset("dataDriven", data=bool(args.dataDriven))
        f.create_dataset("noDeq", data=bool(args.noDeq))
        f.create_dataset("skipHMC", data=bool(args.skipHMC))
        f.create_dataset("gradClip", data=float(args.gradClip))
        f.create_dataset("gradAccum", data=int(args.gradAccum))
        f.create_dataset("alpha", data=float(args.alpha))
        f.create_dataset("dataPath", data=np.string_(args.dataPath or ""))

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
                            flowType=flowType, nsfBins=nsfBins, nsfBound=nsfBound,
                            priorType=args.priorType,
                            condPriorSlowStride=args.condPriorSlowStride,
                            condPriorHidden=args.condPriorHidden,
                            priorDf=args.priorDf,
                            hcgScaleShared=bool(args.hcgScaleShared),
                            hcgHidden=args.hcgHidden,
                            hcgDilated=bool(args.hcgDilated),
                            hcgCircular=bool(args.hcgCircular),
                            hcgSharedDilations=(
                                [int(x) for x in args.hcgSharedDilations.split(",")]
                                if args.hcgSharedDilations else None))

#fw = train.symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,sym,device,dtype,name)

loaded_optimizer_state = None
if args.load:
    import os
    import glob
    name = max(glob.iglob(rootFolder+'savings/*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = torch.load(name)
    # New checkpoint format returns Adam state so learnInterface can
    # restore it into the optimizer it creates (avoids Adam warm-up
    # ejecting a converged model — see project_resume_optimizer_state).
    # Legacy checkpoints (bare state_dict) return None, resume behaves
    # as before.
    loaded_optimizer_state = fw.load(saved)
    if loaded_optimizer_state is not None:
        print("  -> also restored optimizer state (Adam m/v moments)")

# Optional: init per-scale HCG CNNs from a trained shared HCG checkpoint.
# Runs BEFORE training starts, and only when scale_shared=0. This is the
# "start from a scale-invariant point" experiment — per-scale then chooses
# to stay similar (scale-invariance confirmed) or differentiate (per-level
# physics beats scale-invariance from this basin).
if args.hcgInitFromShared and not args.load:
    if args.priorType != "hierarchical_conditional_gaussian":
        print(f"[warn] -hcgInitFromShared given but priorType is "
              f"{args.priorType}, skipping.")
    elif args.hcgScaleShared:
        print(f"[warn] -hcgInitFromShared given but -hcgScaleShared=1 "
              f"(per-scale not in use), skipping.")
    else:
        import os
        import glob
        shared_folder = args.hcgInitFromShared
        shared_ckpts = sorted(
            glob.glob(os.path.join(shared_folder, "savings", "*.saving")),
            key=os.path.getctime,
        )
        if not shared_ckpts:
            raise SystemExit(f"-hcgInitFromShared: no *.saving under "
                             f"{shared_folder}/savings/")
        shared_ckpt = shared_ckpts[-1]
        print(f"[hcgInitFromShared] loading shared checkpoint: {shared_ckpt}")
        shared_state = torch.load(shared_ckpt)
        shared_model = (shared_state["model"]
                        if isinstance(shared_state, dict) and "model" in shared_state
                        else shared_state)
        # (1) Copy shared MERA + Symmetrized + prior masks (everything
        # NOT under prior.cnn_shared.*) directly into per-scale model —
        # architectures match on those keys. Without this, only ~40 K
        # CNN params would be initialized; the 10.9 M MERA would start
        # random and dominate initial loss (~2650 vs shared's ~1912).
        non_cnn = {k: v for k, v in shared_model.items()
                   if not k.startswith("prior.cnn_shared.")
                   and not k.startswith("prior.cnns.")}
        missing, unexpected = fw.load_state_dict(non_cnn, strict=False)
        # per-scale prior.cnns.*.* will show up as "missing" here — that's
        # expected; we fill them via init_perscale_from_shared_state next.
        print(f"[hcgInitFromShared] MERA + Symmetrized state loaded "
              f"({len(non_cnn)} tensors); "
              f"{len(missing)} missing (mostly prior.cnns.*.* — filled next), "
              f"{len(unexpected)} unexpected.")
        # (2) Expand shared CNN weights to every per-scale CNN
        prior_instance = fw.flow.prior if hasattr(fw, "flow") else fw.prior
        prior_instance.init_perscale_from_shared_state(shared_model)
        # (3) Also expand Adam state if the shared checkpoint carries it.
        # Without this, per-scale would create fresh Adam m=v=0 → the first
        # Adam step (bias-corrected to lr·sign(g) even at LOSS minimum)
        # kicks 10.9 M MERA params out of the shared basin → LOSS drifts up
        # 10-20 nat and per-scale can't recover. Preserving shared's Adam
        # moments for MERA (identical arch) and duplicating them to each
        # per-scale CNN sidesteps that warm-up.
        if isinstance(shared_state, dict) and "optimizer" in shared_state:
            shared_opt = shared_state["optimizer"]
            # Build a fresh matching shared HCG so we can enumerate its
            # trainable params in the SAME order the shared optimizer saw
            # them. IMPORTANT: filter by requires_grad — RNVP maskList
            # buffers show up in named_parameters() but are frozen and
            # therefore absent from the Adam optimizer's param_group.
            # Mixing them into positions ⇒ optimizer.load_state_dict()
            # rejects the group with a size-mismatch ValueError.
            shared_fw_tmp = train.symmetryMERAInit(
                L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, name+"_tmp_shared_for_opt",
                depthMERA=depthMERA, weightTying=weightTying, haarPrior=haarPrior,
                flowType=flowType, nsfBins=nsfBins, nsfBound=nsfBound,
                priorType="hierarchical_conditional_gaussian",
                hcgScaleShared=True, hcgHidden=args.hcgHidden,
                hcgDilated=bool(args.hcgDilated), hcgCircular=bool(args.hcgCircular),
                hcgSharedDilations=None,
            )
            shared_trainable = [(n, p) for n, p in shared_fw_tmp.named_parameters() if p.requires_grad]
            shared_name_to_pos = {n: p for p, (n, _) in enumerate(shared_trainable)}
            del shared_fw_tmp

            perscale_trainable = [(n, p) for n, p in fw.named_parameters() if p.requires_grad]

            perscale_state = {}
            n_mera = 0
            n_cnn = 0
            for p_pos, (p_name, _) in enumerate(perscale_trainable):
                if p_name.startswith("prior.cnns."):
                    # e.g., "prior.cnns.0.0.weight" -> "prior.cnn_shared.0.weight"
                    _, _, _, tail = p_name.split(".", 3)
                    shared_equiv = "prior.cnn_shared." + tail
                    n_cnn += 1
                else:
                    shared_equiv = p_name
                    n_mera += 1
                s_pos = shared_name_to_pos.get(shared_equiv)
                if s_pos is None or s_pos not in shared_opt["state"]:
                    continue
                src = shared_opt["state"][s_pos]
                new_entry = {}
                for k, v in src.items():
                    new_entry[k] = v.clone() if isinstance(v, torch.Tensor) else v
                perscale_state[p_pos] = new_entry

            loaded_optimizer_state = {
                "state": perscale_state,
                "param_groups": [{
                    **shared_opt["param_groups"][0],
                    "params": list(range(len(perscale_trainable))),
                }],
            }
            print(f"[hcgInitFromShared] Adam state expanded: "
                  f"{n_mera} MERA + {n_cnn} CNN trainable params → "
                  f"{len(perscale_state)}/{len(perscale_trainable)} state entries populated "
                  f"(CNN entries duplicated from shared CNN)")

# Optional: warm-start from a smaller-L champion (stride-aligned transfer).
# Copies MERA blocks 0..min(N_src, N_tgt)-1 and HCG per-scale CNNs by
# matched stride. Only fires when the source is a per-scale HCG checkpoint
# and target uses per-scale HCG too. See train/transfer.py.
if args.loadFromSmallerL and not args.load:
    if args.priorType != "hierarchical_conditional_gaussian":
        print(f"[warn] -loadFromSmallerL given but priorType is "
              f"{args.priorType}; skipping.")
    elif args.hcgScaleShared:
        print("[warn] -loadFromSmallerL given but -hcgScaleShared=1 "
              "(target has no per-scale CNNs to warm-start); skipping.")
    elif not args.loadFromSmallerLStrides:
        raise SystemExit("-loadFromSmallerL requires -loadFromSmallerLStrides "
                         "(comma-separated source strides, e.g. '16,8,4,2,1' for L=32)")
    else:
        from train.transfer import transfer_from_smaller_L
        src_strides = [int(s) for s in args.loadFromSmallerLStrides.split(",")]
        transfer_from_smaller_L(fw, args.loadFromSmallerL,
                                src_strides=src_strides, device=device,
                                components=args.loadFromSmallerLComponents)

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
    volumePreservingWeight=args.volumePreservingWeight,
    volumePreservingPerLayer=bool(args.volumePreservingPerLayer),
    pathGrad=args.pathGrad,
    scaleLoss=args.scaleLoss,
    physRegWeightChi=args.physRegWeightChi,
    physRegWeightU4=args.physRegWeightU4,
    physRegBatch=args.physRegBatch,
    physRegTargetChi=args.physRegTargetChi,
    physRegTargetU4=args.physRegTargetU4,
    bf16=args.bf16,
    cosineAnneal=args.cosineAnneal, cosineEtaMin=args.cosineEtaMin,
    gradAccum=args.gradAccum,
    optimizerState=loaded_optimizer_state,
)
#LOSS,ZACC,ZOBS,XACC,XOBS = train.learnInterface(target,fw,batch,epochs,save=True,saveSteps = savePeriod,savePath=rootFolder,measureFn = measure)
