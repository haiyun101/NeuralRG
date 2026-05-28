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

def symmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,symmetryList,device,dtype,name = None, channel = 1, depthMERA = None,  weightTying=False, haarPrior=False, flowType="rnvp", nsfBins=8, nsfBound=5.0):
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
            d = flow.save()
            torch.save(d,savePath+flow.name+".saving")

    return LOSS,ACC,OBS


def learnInterface(source, flow, batchSize, epochs, lr=1e-3, save=True, saveSteps=10, savePath=None, keepSavings=3, weight_decay=0.001, adaptivelr=False, HMCsteps=10, HMCthermal=10, HMCepsilon=0.2, measureFn=None, alpha=1.0, skipHMC=True, dataDriven=False, dataPath=None, targetT=None, noDeq=False, jsLoss=False, jsLambda=0.5, jsMemOpt=False, entropyBeta=0.0):

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
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)
        data_iterator = iter(dataloader)
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

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ZACC = []
    XACC = []
    ZOBS = []
    XOBS = []
    ENERGY = []   # <--- 记录能量
    ENTROPY = []  # <--- 记录微分熵
    MAG_ABS = []  # per-epoch <|M|>_q -- bridge integrity proxy
    MAG_VAR = []  # per-epoch Var(M)_q -- susceptibility proxy

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
            log_prob = flow.logProbability(x_std) - log_jac_std

            # Record Energy and Entropy for the MCMC dataset
            # (source is defined on the original-scale x — do NOT standardize here)
            energy_val = -source.logProbability(x).mean().item()
            entropy_val = -log_prob.mean().item()

            lossorigin = -log_prob
            lossstd = lossorigin.std()
            loss = lossorigin.mean()

            if alpha > 0:
                # -log_jac_std cancels in the difference, but keep it explicit
                # so both terms are on the same (physical) scale.
                log_prob_sym = flow.logProbability(-x_std) - log_jac_std
                loss += alpha * (log_prob.mean() - log_prob_sym.mean())**2

            if entropyBeta > 0:
                # Entropy regularizer: subtract beta*H(q) so we maximize H(q).
                # Sequential backward (like jsMemOpt) to keep peak memory ~1 graph
                # at a time instead of MLE-graph + sampling-graph simultaneously.
                # The MLE graph is freed before sampling begins.
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
        skip_outer_backward = (jsLoss and jsMemOpt) or (dataDriven and entropyBeta > 0)
        if not skip_outer_backward:
            flow.zero_grad()
            loss.backward()
        optimizer.step()
        
        if adaptivelr:
            scheduler.step()

        if jsLoss or not dataDriven:
            del sampleLogProbability

        if jsLoss:
            print("epoch:", epoch, "L_js:", loss.item(),
                  "L_rev:", rev_term.item(), "L_fwd:", fwd_term.item(),
                  "+/-", lossstd.item())
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
                    f.create_dataset("ZACC",data=np.array(ZACC))
                    f.create_dataset("ZOBS",data=np.array(ZOBS))
                    f.create_dataset("XACC",data=np.array(XACC))
                    f.create_dataset("XOBS",data=np.array(XOBS))
                d = flow.save()
                torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
                # cleanSaving(epoch)

        del x

    return LOSS,ZACC,ZOBS,XACC,XOBS
