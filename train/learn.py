import torch
from torch import nn
import h5py
import json
import os
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

import glob # Add for mcmc
from torch.utils.data import DataLoader, TensorDataset # Add for mcmc
import re

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

def _compute_scale_invariance_loss(intermediates):
    """Multi-scale loss term: scale-invariance witness across MERA scales.

    For each adjacent scale pair (s, s+1) we take the *same lattice
    positions* under stride 2^(s+1) — these are the kept-coarse slots
    of scale s+1. Their values at scale s come from y_s after the slow
    mode has been processed at scales 0..s; at scale s+1 they've also
    been touched by scale s+1's blocks. After per-sample z-scoring,
    the two should match if the flow obeys approximate scale
    invariance. Mismatch is penalized via MSE.

    Per-sample z-scoring (not batch-wise) means the loss cannot be
    minimised by simply rescaling the latent — the flow must produce
    structurally similar fields at adjacent scales.

    Returns a 0-d tensor (mean across all scale pairs). Backprop-safe.
    """
    if len(intermediates) < 2:
        return intermediates[0].new_zeros(())
    loss = intermediates[0].new_zeros(())
    pair_count = 0
    for s in range(len(intermediates) - 1):
        y_s = intermediates[s]
        y_sp1 = intermediates[s + 1]
        stride = 2 ** (s + 1)  # kept-coarse slot stride at scale s+1
        if y_s.shape[-1] % stride != 0 or y_s.shape[-2] % stride != 0:
            # MERA depth bounds the strides we can probe; skip past it.
            break
        a = y_s[..., ::stride, ::stride]
        b = y_sp1[..., ::stride, ::stride]
        # per-sample z-score across the spatial dims; keep channel separate
        a_mean = a.mean(dim=(-2, -1), keepdim=True)
        b_mean = b.mean(dim=(-2, -1), keepdim=True)
        a_std = a.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        b_std = b.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        a_z = (a - a_mean) / a_std
        b_z = (b - b_mean) / b_std
        loss = loss + (a_z - b_z).pow(2).mean()
        pair_count += 1
    if pair_count > 0:
        loss = loss / pair_count
    return loss


def symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,symmetryList,device,dtype,name = None, channel = 1, depthMERA = None,  weightTying=False, haarPrior=False, flowType="rnvp", nsfBins=8, nsfBound=5.0, priorType="gaussian", condPriorSlowStride=-1, condPriorHidden=32, priorDf=4.0, hcgScaleShared=True, hcgHidden=32, hcgDilated=True, hcgCircular=True, hcgSharedDilations=None):
    # Latent prior selection. The 'conditional_gaussian' option is
    # scheme A from analyzers/rg_fixed_point/improvements_zh.md: relaxes
    # the implicit demand that fast modes be marginally independent
    # N(0,1), which is the Wilson-Gaussian-FP geometry incompatible with
    # Ising T_c. Default 'gaussian' reproduces the original baseline.
    if priorType == "gaussian":
        s = source.Gaussian([channel]+[L]*d)
    elif priorType == "conditional_gaussian":
        # Default heuristic: stride = L//4 gives a 4x4 slow grid for L=16,
        # 8x8 for L=32 etc. — enough slow sites for the CNN to learn a
        # meaningful conditional, while leaving >75% of the field as
        # genuine fast modes whose marginal can be widened by the prior.
        slow_stride = condPriorSlowStride if condPriorSlowStride > 0 else max(2, L // 4)
        print(f">>> Using Conditional Gaussian prior  "
              f"(slow_stride={slow_stride}, slow grid {L//slow_stride}x{L//slow_stride}, "
              f"cnn hidden={condPriorHidden})")
        s = source.ConditionalGaussian([channel]+[L]*d,
                                       slow_stride=slow_stride,
                                       n_hidden=condPriorHidden)
    elif priorType == "hierarchical_conditional_gaussian":
        # Path C of prior_offload_analysis_zh.md: replace single-CNN i2 with a
        # hierarchical Gaussian prior over z, decomposed by successive strides
        # [L/2, L/4, ..., 1]. Each level's conditional Gaussian is scored by a
        # CNN (shared across levels by default → scale-invariant conditional
        # whitening as a direct architectural RG prior). Fixes i2's edge bug
        # by using circular padding by default.
        print(f">>> Using Hierarchical Conditional Gaussian prior "
              f"(scale_shared={hcgScaleShared}, hidden={hcgHidden}, "
              f"dilated={hcgDilated}, circular_pad={hcgCircular}, "
              f"shared_dilations={hcgSharedDilations})")
        s = source.HierarchicalConditionalGaussian(
            [channel]+[L]*d,
            n_hidden=hcgHidden,
            scale_shared=hcgScaleShared,
            use_circular_padding=hcgCircular,
            dilated_conv=hcgDilated,
            shared_dilations=hcgSharedDilations,
        )
    elif priorType == "studentT":
        print(f">>> Using Student-t prior  (df={priorDf}, "
              f"marginal sigma so var=1)")
        s = source.StudentT([channel]+[L]*d, df=priorDf)
    else:
        raise ValueError(f"unknown priorType: {priorType}")

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

    flowType = flowType.lower()
    assert flowType in ("rnvp", "nsf"), f"unknown flowType: {flowType}"

    # MLP dim list -- last entry depends on coupling type
    nsf_out = coreSize * (3 * nsfBins - 1)
    dimList_rnvp = [coreSize] + [nhidden]*nmlp + [coreSize]
    dimList_nsf  = [coreSize] + [nhidden]*nmlp + [nsf_out]

    def _make_rnvp_block(mask):
        tList = [utils.SimpleMLPreshape(dimList_rnvp,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)]
        sList = [utils.SimpleMLPreshape(dimList_rnvp,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(coreSize)]) for _ in range(nlayers)]
        return flow.RNVP(mask, tList, sList)

    def _make_nsf_block(mask):
        # NSF conditioner outputs are spline parameters; no tanh/scale wrapper.
        # Reshape input from (B, channel, 2, 2) to (B, coreSize) -- the existing
        # SimpleMLPreshape only works when input.shape == output.shape, so build
        # a Flatten+MLP+Identity-output stack ourselves.
        #
        # Identity init: zero the final linear weight and set its bias so
        # softmax(widths)=softmax(heights)=uniform and softplus(derivs)+min ≈ 1.
        # This makes every spline start as the identity on [-B,B], matching
        # RNVP's zero-init convention and preventing wide-MLP divergence.
        d_init = math.log(math.exp(1.0 - flow.nsf.DEFAULT_MIN_DERIV) - 1.0)
        per_spline = torch.zeros(3 * nsfBins - 1)
        per_spline[2 * nsfBins:] = d_init        # raw_derivs entries
        bias_pattern = per_spline.repeat(coreSize)

        netList = []
        for _ in range(nlayers):
            net = nn.Sequential(
                nn.Flatten(),
                *[layer for k in range(nmlp + 1)
                  for layer in [nn.Linear(dimList_nsf[k], dimList_nsf[k+1]),
                                nn.ELU() if k < nmlp else nn.Identity()]],
            )
            # zero-init final Linear weight + identity-init bias
            final = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                final.weight.zero_()
                final.bias.copy_(bias_pattern)
            netList.append(net)
        return flow.NSFCoupling(mask, netList, num_bins=nsfBins, bound=nsfBound)

    make_block = _make_nsf_block if flowType == "nsf" else _make_rnvp_block

    layers = []
    if weightTying:
        print(f">>> Using Physical Prior: Weight Tying (Scale Invariance)  [flowType={flowType}]")
        shared = make_block(MaskList[0])
        if haarPrior:
            print(">>> Using Physical Prior: Haar Wavelet (Majority Vote)")
            shared = HaarRNVP(shared)
        layers = [shared for _ in range(depth)]
    else:
        if flowType == "nsf":
            print(f">>> Using NSF coupling (K={nsfBins} bins, bound={nsfBound})")
        for n in range(depth):
            block = make_block(MaskList[n])
            if haarPrior:
                if n == 0:
                    print(">>> Using Physical Prior: Haar Wavelet (Majority Vote)")
                block = HaarRNVP(block)
            layers.append(block)
            
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
            d = flow.save(optimizer=optimizer)
            torch.save(d,savePath+flow.name+".saving")

    return LOSS,ACC,OBS


def learnInterface(source, flow, batchSize, epochs, lr=1e-3, save=True, saveSteps=10, savePath=None, keepSavings=3, weight_decay=0.001, adaptivelr=False, HMCsteps=10, HMCthermal=10, HMCepsilon=0.2, measureFn=None, alpha=1.0, skipHMC=True, dataDriven=False, dataPath=None, targetT=None, noDeq=False, jsLoss=False, jsLambda=0.5, jsMemOpt=False, entropyBeta=0.0, gradClip=0.0, bridgeWeight=0.0, bridgeThresh=0.5, volumePreservingWeight=0.0, volumePreservingPerLayer=False, pathGrad=False, scaleLoss=0.0, physRegWeightChi=0.0, physRegWeightU4=0.0, physRegBatch=128, physRegTargetChi=float("nan"), physRegTargetU4=float("nan"), bf16=False, cosineAnneal=False, cosineEtaMin=None, gradAccum=1, optimizerState=None):
    # gradAccum: number of micro-batches to accumulate gradients over before
    # calling optimizer.step(). Splits batchSize into gradAccum micro-batches
    # of size (batchSize // gradAccum) each, so effective batch is preserved.
    # Default 1 = no accumulation, behavior identical to old runs. Only the
    # dataDriven branch implements accumulation (which is what we currently
    # need; other modes will fall back to gradAccum=1).
    gradAccum = max(1, int(gradAccum))
    if gradAccum > 1 and not dataDriven:
        print(f"  [warning] gradAccum={gradAccum} requested but mode is not dataDriven; ignoring.")
        gradAccum = 1
    microBatch = batchSize // gradAccum
    if gradAccum > 1:
        assert batchSize % gradAccum == 0, \
            f"batchSize={batchSize} not divisible by gradAccum={gradAccum}"
        print(f"  [gradAccum] eff_batch={batchSize} = micro_batch={microBatch} x K={gradAccum}")

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
        
    # ==============================================================
    # 1. DATA-DRIVEN DATA LOADING BLOCK
    # ==============================================================
    L_dim = int(flow.prior.sample(1).shape[-1])
    needs_data = dataDriven or jsLoss
    if needs_data:
        if dataPath is None:
            search_pattern = f"./data/mcmc_data/mcmc_wolff_L{L_dim}_T*_N*.pt"
            all_files = glob.glob(search_pattern)

            found_files = []
            for file in all_files:
                # Extract the T value using regex
                match = re.search(r'_T([\d\.]+)_N', file)
                if match:
                    file_T = float(match.group(1))
                    # Use a small tolerance for floating point comparison (e.g., 1e-5)
                    if abs(file_T - targetT) < 1e-5:
                        found_files.append(file)

            if not found_files:
                raise FileNotFoundError(f"Could not automatically find data matching {search_pattern}")
            dataPath = found_files[0]
            print(f"Auto-selected MCMC dataset: {dataPath}")

        if jsLoss:
            print(f"Initializing JS-like (symmetrized-KL) Training  |  lambda = {jsLambda}")
        else:
            print("Initializing Data-Driven MLE Training...")
        mcmc_data = torch.load(dataPath)
        # HS samples have std ~3.5 (range ±11) while the RNVP coupling MLPs were
        # designed for ~unit-scale inputs; large inputs make the log-Jacobian
        # numerically unreliable (loss can fall below the entropy floor H(p_HS)).
        # Standardize the flow input by a single global scalar σ and undo it
        # exactly via the change-of-variables Jacobian:
        #     log q_X(x) = log q(x/σ) - N·logσ
        # The correction is constant in the flow parameters, so training
        # dynamics are unchanged, but LOSS / ENTROPY stay directly comparable
        # to the theory entropy. For discrete data σ≈1 → effectively a no-op.
        if jsLoss:
            # JS mode mixes reverse-KL (flow samples physical x) with
            # forward-KL (data x). Both terms must share the flow's coordinate
            # system, so we do NOT standardize for JS — the flow trains in
            # physical x throughout (matching the reverse-KL convention).
            data_std = 1.0
            n_dim = int(mcmc_data[0].numel())
            log_jac_std = 0.0
            print(f"JS input scale: sigma=1.0 (physical x, no standardization), N={n_dim}")
        else:
            data_std = float(mcmc_data.std())
            n_dim = int(mcmc_data[0].numel())
            log_jac_std = n_dim * float(np.log(data_std))
            print(f"Data-driven input standardization: sigma={data_std:.6f}, "
                  f"N={n_dim}, N*log(sigma)={log_jac_std:.4f} "
                  f"(loss corrected to remain comparable to H(p_HS))")
        dataset = TensorDataset(mcmc_data)
        dataloader = DataLoader(dataset, batch_size=microBatch, shuffle=True, drop_last=True)
        data_iterator = iter(dataloader)

        # ── Physical-observable regularizer setup ───────────────────────
        # Auto-compute chi/U4 targets from data if user didn't provide.
        # Convention: use soft spin s = tanh(x_physical), M = mean(s) per sample,
        #   chi = N * (⟨M²⟩ − ⟨|M|⟩²),  U4 = 1 − ⟨M⁴⟩ / (3⟨M²⟩²)
        # Physical x = data_std * x_standardized (undo -dataDriven scaling).
        # We compute on the FULL data batch (or a big subset) for accurate GT.
        _need_phys_reg = (physRegWeightChi > 0.0 or physRegWeightU4 > 0.0)
        if _need_phys_reg or not (physRegTargetChi != physRegTargetChi):  # nan check via self-ineq
            with torch.no_grad():
                _n_data_for_gt = min(len(mcmc_data), 5000)
                _idx = torch.randperm(len(mcmc_data))[:_n_data_for_gt]
                _x_gt = mcmc_data[_idx].float()   # already physical (not standardized)
                if _x_gt.dim() == 3:
                    _x_gt = _x_gt.unsqueeze(1)
                _s = torch.tanh(_x_gt)             # soft spin ≈ sign(x) for |x| >> 1
                _M = _s.reshape(_s.shape[0], -1).mean(dim=-1)   # per-sample magnetization
                _N_sites = _s.shape[-1] * _s.shape[-2]
                _chi_data = _N_sites * (_M.pow(2).mean() - _M.abs().mean().pow(2))
                _M2 = _M.pow(2).mean(); _M4 = _M.pow(4).mean()
                _U4_data = 1.0 - _M4 / (3 * _M2.pow(2) + 1e-12)
            print(f"  [phys-reg] data-derived targets from N={_n_data_for_gt}: "
                  f"chi = {_chi_data.item():.3f}, U4 = {_U4_data.item():.4f}")
        # Resolve targets: user override wins if not NaN
        if physRegTargetChi != physRegTargetChi:   # NaN
            physRegTargetChi = float(_chi_data.item()) if _need_phys_reg else float("nan")
        if physRegTargetU4 != physRegTargetU4:
            physRegTargetU4 = float(_U4_data.item()) if _need_phys_reg else float("nan")
        if _need_phys_reg:
            print(f"  [phys-reg] using targets: chi = {physRegTargetChi:.3f}, "
                  f"U4 = {physRegTargetU4:.4f}   "
                  f"(λ_chi={physRegWeightChi}, λ_U4={physRegWeightU4}, batch={physRegBatch})")
    else:
        # Reverse-KL: if we're resuming from a forward-KL run that used input
        # standardization, the flow expects standardized inputs and returns
        # standardized samples u from .sample(). We must convert u -> physical
        # x = sigma*u before evaluating the action source.logProbability(x).
        # Detect this by reading any existing flow_input_sigma.json. If absent
        # or sigma=1.0, this is a fresh reverse-KL run (no scaling needed).
        prior_sigma = 1.0
        try:
            _sig_path = os.path.join(savePath, "flow_input_sigma.json")
            if os.path.exists(_sig_path):
                with open(_sig_path) as _f:
                    prior_sigma = float(json.load(_f).get("sigma", 1.0))
        except Exception:
            pass
        if abs(prior_sigma - 1.0) > 1e-6:
            print(f"Initializing Reverse-KL Finetune (Phase 2)  |  "
                  f"inherited sigma={prior_sigma:.6f} from prior phase")
            data_std = prior_sigma
        else:
            print("Initializing Energy-Based Reverse-KL Training...")
            data_std = 1.0

    # Record the flow-input scale so post-hoc tools (analyzers/flow_sample_diagnostic.py)
    # know whether flow.sample() returns standardized u=x/sigma or physical x.
    # sigma=1.0 for fresh reverse-KL; sigma=std(data) for data-driven or finetune.
    try:
        with open(os.path.join(savePath, "flow_input_sigma.json"), "w") as _f:
            json.dump({"sigma": float(data_std), "dataDriven": bool(dataDriven)}, _f)
    except Exception as _e:
        print(f"Warning: could not write flow_input_sigma.json: {_e}")
    # ==============================================================

    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizerState is not None:
        optimizer.load_state_dict(optimizerState)
        print("[resume] restored Adam state — skipping ~500-epoch warm-up")

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
    elif cosineAnneal:
        eta_min = cosineEtaMin if cosineEtaMin is not None else lr * 0.01
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        print(f"[cosineAnneal] T_max={epochs} eta_min={eta_min:.2e} (lr {lr:.2e} → {eta_min:.2e})")

    LOSS = []
    ZACC = []
    XACC = []
    ZOBS = []
    XOBS = []
    ENERGY = []   # <--- 记录能量
    ENTROPY = []  # <--- 记录微分熵
    MAG_ABS = []  # per-epoch <|M|>_q -- bridge integrity proxy
    MAG_VAR = []  # per-epoch Var(M)_q -- susceptibility proxy
    BRIDGE_FRAC = []  # per-epoch fraction of batch with |M_i| < bridgeThresh (bridge upweighting diagnostic)
    SCALE_LOSS = []   # per-epoch raw multi-scale loss (BEFORE lambda_scale scaling) for ablation; nan when scaleLoss==0

    # Resolve the inner MERA for multi-scale loss intermediates.
    # When Symmetrized wraps the MERA, scale loss is computed on the
    # un-symmetrized branch only (one extra forward pass per step). This
    # keeps the cost predictable and the loss term interpretable.
    _inner_mera = getattr(flow, "flow", flow) if (scaleLoss > 0 or volumePreservingWeight > 0) else None

    z_ = flow.prior.sample(batchSize)
    x_ = flow.prior.sample(batchSize)

    L = int(x_.shape[-1]**0.5)

    for epoch in range(epochs):
        
        # ==============================================================
        # 2. TOGGLED TRAINING LOOP (Reverse-KL / Forward-KL / JS-like)
        # ==============================================================
        if jsLoss:
            # --- JS-like symmetrized KL ---
            # loss = jsLambda * KL(q||p) + (1-jsLambda) * KL(p||q)
            # KL(q||p) estimator: E_q[log q - log p_unnorm]   (lnZ_c constant dropped)
            # KL(p||q) estimator: E_p[-log q]                 (H(p) constant dropped)
            # Both terms feed/score the flow in physical x (no standardization),
            # so log q(x) is consistent across the two batches.
            #
            # Two backward strategies:
            #   jsMemOpt=False (default): build the combined loss tensor, single
            #     backward at the bottom of the loop. Both graphs alive simultaneously
            #     -> ~2x peak GPU memory of a single-direction step.
            #   jsMemOpt=True : backward the rev half, free its graph, then build
            #     and backward the fwd half. Gradients accumulate into params.
            #     -> peak memory is one graph at a time (~halved).

            # Reverse-KL term (mode-seeking) -- flow samples
            x_q, sampleLogProbability = flow.sample(batchSize)
            rev_per_sample = sampleLogProbability - source.logProbability(x_q)
            rev_term = rev_per_sample.mean()

            energy_val = -source.logProbability(x_q).mean().item()
            entropy_val = -sampleLogProbability.mean().item()
            lossorigin = rev_per_sample      # F-column = variational free energy
            lossstd = rev_per_sample.std()

            if jsMemOpt:
                # Backward the reverse half immediately to free its graph
                flow.zero_grad()
                (jsLambda * rev_term).backward()
                rev_val = rev_term.detach()

                # Now the forward half (fresh graph, reverse graph already freed)
                try:
                    x_real, = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x_real, = next(data_iterator)
                x_real = x_real.to(device=x_q.device, dtype=x_q.dtype)
                if not noDeq:
                    x_real = x_real + (torch.rand_like(x_real) - 0.5) * 0.2
                log_q_p = flow.logProbability(x_real)
                fwd_term = (-log_q_p).mean()
                ((1.0 - jsLambda) * fwd_term).backward()
                fwd_val = fwd_term.detach()

                # Combined scalar (no graph) for logging
                loss = jsLambda * rev_val + (1.0 - jsLambda) * fwd_val
                loss_already_backward = True
            else:
                # Forward-KL term (mass-covering) -- HS data
                try:
                    x_real, = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x_real, = next(data_iterator)
                x_real = x_real.to(device=x_q.device, dtype=x_q.dtype)
                if not noDeq:
                    x_real = x_real + (torch.rand_like(x_real) - 0.5) * 0.2
                log_q_p = flow.logProbability(x_real)
                fwd_term = (-log_q_p).mean()

                loss = jsLambda * rev_term + (1.0 - jsLambda) * fwd_term
                loss_already_backward = False

            x = x_q     # alias for the per-epoch teardown (`del x`) below
            del x_real

        elif dataDriven:
            # --- Data-Driven Forward KL (MLE) ---
            # Optional bf16 autocast: wraps forward passes in bfloat16 for
            # ~2x speedup on A100 Tensor Cores. No GradScaler needed since
            # bf16 has fp32-equivalent dynamic range. Gradients + optimizer
            # stay in fp32. Autocast auto-detects backward pass — outside
            # the `with` block is safe (all our .backward() calls are).
            from contextlib import nullcontext
            _amp_ctx = (torch.cuda.amp.autocast(dtype=torch.bfloat16)
                        if bf16 and str(x_.device).startswith("cuda")
                        else nullcontext())
            # Gradient-accumulation guard: entropyBeta path does its own
            # flow.zero_grad() inside the loss body, which would wipe out
            # accumulated grads. The two are not jointly supported.
            if gradAccum > 1 and entropyBeta > 0:
                raise ValueError("gradAccum > 1 is incompatible with entropyBeta > 0")
            # For gradAccum > 1, we zero grads ONCE here and accumulate
            # across K micro-batches before the outer optimizer.step().
            # K=1 path is unchanged: zero_grad still happens later (line ~634).
            if gradAccum > 1:
                flow.zero_grad()
                _accum_loss_sum = 0.0
                _accum_lossstd_sq_sum = 0.0
                _accum_energy_sum = 0.0
                _accum_entropy_sum = 0.0
                _accum_scale_loss_sum = 0.0
                _accum_bridge_frac_sum = 0.0
            # When K=1, the for-loop runs once and behavior is identical to
            # the single-pass baseline.
            for _accum_k in range(gradAccum):
                try:
                    x_real, = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x_real, = next(data_iterator)

                x_real = x_real.to(device=x_.device, dtype=x_.dtype)

                if noDeq:
                    x = x_real
                else:
                    # Dequantization noise to smooth the discrete states
                    noise = (torch.rand_like(x_real) - 0.5) * 0.2
                    x = x_real + noise

                # Feed the flow ~unit-scale inputs; recover the physical log-density
                # on the original x via the standardization Jacobian (constant term,
                # so gradients/training dynamics are identical to training on the
                # standardized data, but the logged loss stays physical).
                x_std = x / data_std
                with _amp_ctx:
                    log_prob = flow.logProbability(x_std) - log_jac_std

                # Record Energy and Entropy for the MCMC dataset
                # (source is defined on the original-scale x — do NOT standardize here)
                energy_val = -source.logProbability(x).mean().item()
                entropy_val = -log_prob.mean().item()  # pure unweighted MLE — comparable across runs

                lossorigin = -log_prob
                lossstd = lossorigin.std()

                if bridgeWeight > 0:
                    # Per-sample magnetization on physical x; bridge = |M_i| < bridgeThresh.
                    # w_i = 1 + bridgeWeight * 1[in bridge], normalized to sum=batch so the
                    # loss magnitude stays comparable to the unweighted case.
                    with torch.no_grad():
                        _M_i = x.reshape(x.shape[0], -1).mean(dim=1)
                        _in_bridge = (_M_i.abs() < bridgeThresh).to(lossorigin.dtype)
                        _w = 1.0 + bridgeWeight * _in_bridge
                        _bridge_frac = float(_in_bridge.mean().item())
                    loss = (_w * lossorigin).sum() / _w.sum()
                else:
                    _bridge_frac = float('nan')
                    loss = lossorigin.mean()

                # Volume-preserving regularizer. Adds λ · mean((log|det J_MERA|)²)
                # to the loss. Under the un-symmetrized MERA, log|det J| per
                # sample tells us how much the flow expands/contracts volume;
                # pushing it toward 0 forces the marginal p(z) to actually
                # equal the HCG prior scale (rescues σ interpretability, see
                # discussion of prior-Jacobian trade-off). Extra cost: one
                # un-symmetrized forward pass.
                _vp_penalty_value = float("nan")
                if volumePreservingWeight > 0:
                    with _amp_ctx:
                        if volumePreservingPerLayer:
                            # Per-layer VP: Σ_block E[(log|det J_block|)²].
                            # Prevents adjacent blocks from cancelling each
                            # other via expand/contract pairs (which the global
                            # VP allows). Each RNVP block must be individually
                            # volume-preserving.
                            _, _per_block = _inner_mera.forward_with_per_block_logjac(x_std)
                            _vp_penalty_raw = sum(lj.pow(2).mean() for lj in _per_block)
                        else:
                            # Global VP: E[(Σ log|det J_block|)²]. Allows cancellation.
                            _, _logjac_mera = _inner_mera.forward(x_std)
                            _vp_penalty_raw = _logjac_mera.pow(2).mean()
                    loss = loss + volumePreservingWeight * _vp_penalty_raw
                    _vp_penalty_value = float(_vp_penalty_raw.item())

                # Multi-scale loss (III.1 in improvements_zh.md). Calls the
                # un-symmetrized MERA's forward_with_intermediates and
                # penalizes mismatch between adjacent scales' kept-coarse
                # subsets after per-sample z-scoring. Default coefficient
                # 0 disables. Extra cost = one un-symmetrized forward.
                if scaleLoss > 0:
                    with _amp_ctx:
                        _, _, _intermediates = _inner_mera.forward_with_intermediates(x_std)
                        _scale_loss_raw = _compute_scale_invariance_loss(_intermediates)
                    loss = loss + scaleLoss * _scale_loss_raw
                    _scale_loss_value = float(_scale_loss_raw.item())
                else:
                    _scale_loss_value = float('nan')

                # Physical-observable regularizer: match χ and U₄ of flow
                # samples to data-derived targets. Adds:
                #   λ_χ · (χ_flow − χ_target)²   +   λ_U · (U₄_flow − U₄_target)²
                # Sampling is differentiable so gradient flows through.
                # Un-standardize flow output to physical x (undo dataDriven
                # scaling) before applying tanh spin proxy. Costs 1 flow.sample
                # of size physRegBatch per micro-step.
                _phys_chi_value = float("nan"); _phys_u4_value = float("nan")
                if physRegWeightChi > 0.0 or physRegWeightU4 > 0.0:
                    with _amp_ctx:
                        _u_flow, _ = flow.sample(physRegBatch)  # standardized samples
                    _x_flow_phys = _u_flow * data_std        # undo dataDriven scaling
                    _s_soft = torch.tanh(_x_flow_phys)       # soft spin, ≈ sign for |x|≫1
                    _M_flow = _s_soft.reshape(_s_soft.shape[0], -1).mean(dim=-1)  # (B,)
                    _N_sites = _s_soft.shape[-1] * _s_soft.shape[-2]
                    _chi_flow = _N_sites * (_M_flow.pow(2).mean() - _M_flow.abs().mean().pow(2))
                    _M2 = _M_flow.pow(2).mean(); _M4 = _M_flow.pow(4).mean()
                    _U4_flow = 1.0 - _M4 / (3 * _M2.pow(2) + 1e-12)
                    _phys_chi_value = float(_chi_flow.item())
                    _phys_u4_value = float(_U4_flow.item())
                    if physRegWeightChi > 0.0:
                        loss = loss + physRegWeightChi * (_chi_flow - physRegTargetChi).pow(2)
                    if physRegWeightU4 > 0.0:
                        loss = loss + physRegWeightU4 * (_U4_flow - physRegTargetU4).pow(2)

                if alpha > 0:
                    # -log_jac_std cancels in the difference, but keep it explicit
                    # so both terms are on the same (physical) scale.
                    with _amp_ctx:
                        log_prob_sym = flow.logProbability(-x_std) - log_jac_std
                    loss += alpha * (log_prob.mean() - log_prob_sym.mean())**2

                if entropyBeta > 0:
                    # Entropy regularizer: subtract beta*H(q) so we maximize H(q).
                    # Sequential backward (like jsMemOpt) to keep peak memory ~1 graph
                    # at a time instead of MLE-graph + sampling-graph simultaneously.
                    # The MLE graph is freed before sampling begins.
                    # (Guarded against gradAccum > 1 at the top of the dataDriven branch.)
                    flow.zero_grad()
                    loss.backward()
                    mle_val = loss.detach()

                    _u_q, _log_q_q = flow.sample(batchSize)
                    _H_q = -_log_q_q.mean()
                    ent_term = -entropyBeta * _H_q
                    ent_term.backward()

                    # Combined scalar for logging; .detach() so no graph
                    loss = mle_val + ent_term.detach()
                    loss_already_backward = True

                if gradAccum > 1:
                    # Accumulate this micro-batch's gradient into .grad.
                    # Divide loss by K so that the accumulated total matches
                    # the gradient of a full (batchSize) batch.
                    (loss / gradAccum).backward()
                    _accum_loss_sum += loss.item()
                    _accum_lossstd_sq_sum += lossstd.item() ** 2
                    _accum_energy_sum += energy_val
                    _accum_entropy_sum += entropy_val
                    _accum_scale_loss_sum += (_scale_loss_value if _scale_loss_value == _scale_loss_value else 0.0)
                    _accum_bridge_frac_sum += (_bridge_frac if _bridge_frac == _bridge_frac else 0.0)

            # End of K-loop (or single pass when gradAccum=1).
            if gradAccum > 1:
                # Build aggregated per-step scalars for logging; gradient
                # has already been accumulated K times in .grad, so the
                # outer dispatch block must skip its zero_grad + backward.
                loss_already_backward = True
                _device = log_prob.device
                _dtype = log_prob.dtype
                loss = torch.tensor(_accum_loss_sum / gradAccum, device=_device, dtype=_dtype)
                lossstd = torch.tensor((_accum_lossstd_sq_sum / gradAccum) ** 0.5,
                                       device=_device, dtype=_dtype)
                energy_val = _accum_energy_sum / gradAccum
                entropy_val = _accum_entropy_sum / gradAccum
                _scale_loss_value = _accum_scale_loss_sum / gradAccum if scaleLoss > 0 else float('nan')
                _bridge_frac = _accum_bridge_frac_sum / gradAccum if bridgeWeight > 0 else float('nan')

        elif pathGrad:
            # --- Reverse-KL Path Gradient (STL / Roeder 2017, Vaitl 2024) ---
            # Standard reparam grad of E_q[log q - log p] has two routes through θ:
            #   (a) implicit, through the sample path x = T_θ(z), and
            #   (b) explicit, through the θ-dependence of log q_θ(·).
            # At q=p the (b) term has mean 0 but variance > 0; dropping it gives
            # the "sticking the landing" property (zero gradient variance at the
            # optimum). Algebraic identity used here:
            #
            #   logq_full     = log q_θ(u)         # grad: (a) + (b)
            #   logq_explicit = log q_θ(u.detach())# grad: (b) only
            #   logq_for_bw   = logq_full - logq_explicit  # value=0, grad=(a) only
            #
            # Cost: one extra inverse pass per step (the second logProbability call).
            u, _ = flow.sample(batchSize)
            x = data_std * u if abs(data_std - 1.0) > 1e-6 else u

            logq_full     = flow.logProbability(u)
            logq_explicit = flow.logProbability(u.detach())
            logq_for_bw   = logq_full - logq_explicit

            logp_target = source.logProbability(x)

            # Logged quantities are the same physical scalars as the standard branch
            energy_val  = -logp_target.mean().item()
            entropy_val = -logq_full.mean().item()
            lossorigin  = (logq_full - logp_target).detach()  # F = ⟨log q - log p⟩, value only
            lossstd     = lossorigin.std()

            # Loss tensor whose .backward() yields the path-only gradient.
            # Symmetry penalty same form as standard branch — STL applies the
            # path-only filter to the main KL term; the alpha penalty keeps its
            # standard form (it's a regularizer, not the estimator under study).
            loss = (logq_for_bw - logp_target).mean()
            if alpha > 0:
                loss = loss + alpha * (logq_full.mean() - flow.logProbability(-u).mean())**2

        else:
            # --- Original Energy-Based Reverse KL ---
            # When data_std != 1.0 (Phase 2 finetune from a forward-KL flow),
            # flow.sample() returns standardized u; physical x = data_std * u
            # is what the action expects. data_std=1.0 -> u==x (no scaling).
            u, sampleLogProbability = flow.sample(batchSize)
            x = data_std * u if abs(data_std - 1.0) > 1e-6 else u

            # 分离并提取单轮的 Energy 和 Entropy 标量值
            energy_val = -source.logProbability(x).mean().item()
            entropy_val = -sampleLogProbability.mean().item()

            lossorigin = (sampleLogProbability - source.logProbability(x))
            lossstd = lossorigin.std()
            # Symmetry penalty in the flow's native (standardized) coordinate u.
            # -u physical-equivalent is -x, and the flow's logProbability is
            # symmetric in u iff it's symmetric in x for any positive sigma.
            loss = (lossorigin.mean()+alpha*(sampleLogProbability.mean()-flow.logProbability(-u).mean()))
        # ==============================================================

        # In jsMemOpt the JS branch already did flow.zero_grad() and the two
        # backward passes; gradients are accumulated in .grad. The entropy-reg
        # data-driven branch does the same (sequential MLE then -beta*H).
        # Otherwise we do the standard single-pass backward here.
        skip_outer_backward = (jsLoss and jsMemOpt) or (dataDriven and entropyBeta > 0) or (dataDriven and gradAccum > 1)
        if not skip_outer_backward:
            flow.zero_grad()
            loss.backward()
        if gradClip > 0:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), gradClip)
        optimizer.step()
        
        if adaptivelr or cosineAnneal:
            scheduler.step()

        if jsLoss or (not dataDriven and not pathGrad):
            del sampleLogProbability

        if jsLoss:
            print("epoch:", epoch, "L_js:", loss.item(),
                  "L_rev:", rev_term.item(), "L_fwd:", fwd_term.item(),
                  "+/-", lossstd.item())
        else:
            if scaleLoss > 0 and dataDriven:
                print("epoch:",epoch, "L:",loss.item(),"F:",lossorigin.mean().item(),"+/-",lossstd.item(),
                      "L_scale:",_scale_loss_value)
            else:
                print("epoch:",epoch, "L:",loss.item(),"F:",lossorigin.mean().item(),"+/-",lossstd.item())
        del lossorigin

        LOSS.append([loss.item(),lossstd.item()])
        ENERGY.append(energy_val)   # <--- 记录当前轮数能量
        ENTROPY.append(entropy_val) # <--- 记录当前轮数熵

        # Per-epoch magnetization stats from the physical-x batch (sample-side
        # for reverse-KL / JS, data-side for forward-KL data-driven). M is the
        # per-config mean of x components; <|M|> rising sharply signals
        # mode-collapse, Var(M) narrowing signals bridge loss.
        with torch.no_grad():
            _M = x.detach().reshape(x.shape[0], -1).mean(dim=1)
            MAG_ABS.append(float(_M.abs().mean().item()))
            MAG_VAR.append(float(_M.var().item()))
            if dataDriven and bridgeWeight > 0:
                BRIDGE_FRAC.append(_bridge_frac)
            else:
                # keep BRIDGE_FRAC aligned with MAG_ABS for downstream readers
                BRIDGE_FRAC.append(float((_M.abs() < bridgeThresh).float().mean().item()))
            # Per-epoch scale loss (nan when scaleLoss==0) — kept aligned
            # with MAG_ABS so downstream readers can index it the same way.
            if dataDriven and scaleLoss > 0:
                SCALE_LOSS.append(_scale_loss_value)
            else:
                SCALE_LOSS.append(float('nan'))

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
                    f.create_dataset("ENERGY",data=np.array(ENERGY))   # <--- 存入hdf5
                    f.create_dataset("ENTROPY",data=np.array(ENTROPY)) # <--- 存入hdf5
                    f.create_dataset("MAG_ABS",data=np.array(MAG_ABS))
                    f.create_dataset("MAG_VAR",data=np.array(MAG_VAR))
                    f.create_dataset("BRIDGE_FRAC",data=np.array(BRIDGE_FRAC))
                    f.create_dataset("SCALE_LOSS",data=np.array(SCALE_LOSS))
                    f.create_dataset("ZACC",data=np.array(ZACC))
                    f.create_dataset("ZOBS",data=np.array(ZOBS))
                    f.create_dataset("XACC",data=np.array(XACC))
                    f.create_dataset("XOBS",data=np.array(XOBS))
                d = flow.save(optimizer=optimizer)
                torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
                # cleanSaving(epoch)

        del x

    return LOSS,ZACC,ZOBS,XACC,XOBS
