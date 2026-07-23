import torch
from torch import nn

from ..flow import Flow
from .im2col import dispatch,collect


class HierarchyBijector(Flow):
    def __init__(self, kernelShape, indexI, indexJ, layerList,skipCheck = False, prior=None,name = "HierarchyBijector", blocks_per_scale=1):
        super(HierarchyBijector,self).__init__(prior,name)
        if not skipCheck:
            assert len(layerList) == len(indexI)
            assert len(layerList) == len(indexJ)

        self.depth = len(layerList)

        self.kernelShape = kernelShape
        print("kernelShape:",kernelShape)
        self.layerList = torch.nn.ModuleList(layerList)
        self.indexI = indexI
        self.indexJ = indexJ
        # Used by `forward_with_intermediates` to know how many RNVP
        # blocks make up one physical scale (MERA sets it to repeat*2).
        self.blocks_per_scale = blocks_per_scale

    def forward(self,x):
        batchSize = x.shape[0]
        channelSize = x.shape[1]
        forwardLogjac = x.new_zeros(x.shape[0])
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no],self.indexJ[no],x)
            x_,logProbability = self.layerList[no].forward(x_.reshape(-1,channelSize,*self.kernelShape))
            forwardLogjac +=logProbability.reshape(batchSize,-1).sum(1)
            x = collect(self.indexI[no],self.indexJ[no],x,x_)
        return x,forwardLogjac

    def forward_with_per_block_logjac(self, x):
        """Same as ``forward`` but returns per-block log|det J| as a list
        instead of the sum. Used by the per-layer VP penalty:

          L_VP_per_layer = λ · Σ_block  E_data[ (log|det J_block|)² ]

        which forces every RNVP block to be individually volume-preserving,
        vs the default global VP that only penalizes the sum. Fixes the
        "adjacent blocks cancel via expand/contract pairs" pathology that
        allows nr=2 arms to collapse under the standard VP.
        """
        batchSize = x.shape[0]
        channelSize = x.shape[1]
        per_block_logjac = []
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no], self.indexJ[no], x)
            x_, logProbability = self.layerList[no].forward(
                x_.reshape(-1, channelSize, *self.kernelShape))
            per_block_logjac.append(
                logProbability.reshape(batchSize, -1).sum(1))       # (B,)
            x = collect(self.indexI[no], self.indexJ[no], x, x_)
        return x, per_block_logjac

    def forward_with_intermediates(self, x):
        """Same as ``forward`` but also returns per-scale snapshots.

        ``intermediates[s]`` is the LxL field after scale s has been
        applied (i.e. after every ``blocks_per_scale`` consecutive
        RNVP blocks). Used by the multi-scale-loss training path; not
        called on the standard log-prob path so the default ``forward``
        signature stays untouched.
        """
        batchSize = x.shape[0]
        channelSize = x.shape[1]
        forwardLogjac = x.new_zeros(x.shape[0])
        intermediates = []
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no], self.indexJ[no], x)
            x_, logProbability = self.layerList[no].forward(
                x_.reshape(-1, channelSize, *self.kernelShape)
            )
            forwardLogjac += logProbability.reshape(batchSize, -1).sum(1)
            x = collect(self.indexI[no], self.indexJ[no], x, x_)
            if (no + 1) % self.blocks_per_scale == 0:
                intermediates.append(x)
        return x, forwardLogjac, intermediates

    def inverse(self,z):
        batchSize = z.shape[0]
        channelSize = z.shape[1]
        inverseLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            z,z_ = dispatch(self.indexI[no],self.indexJ[no],z)
            z_,logProbability = self.layerList[no].inverse(z_.reshape(-1,channelSize,*self.kernelShape))
            inverseLogjac += logProbability.reshape(batchSize,-1).sum(1)
            z = collect(self.indexI[no],self.indexJ[no],z,z_)
        return z,inverseLogjac