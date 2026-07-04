
import numpy as np
import torch
from torch import nn

class Flow(nn.Module):

    def __init__(self, prior = None,name = "Flow"):
        super(Flow, self).__init__()
        self.name = name
        self.prior = prior

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self,batchSize, prior = None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        z = prior.sample(batchSize)
        logp = prior.logProbability(z)
        x,logp_ = self.inverse(z)
        return x,logp-logp_

    def forward(self,x):
        raise NotImplementedError(str(type(self)))

    def inverse(self,z):
        raise NotImplementedError(str(type(self)))

    def logProbability(self,x):
        z,logp = self.forward(x)
        if self.prior is not None:
            return self.prior.logProbability(z)+logp
        return logp

    def save(self, optimizer=None):
        """Build a checkpoint dict for `torch.save`.

        If ``optimizer`` is passed, the returned dict is
        ``{'model': model_state, 'optimizer': optimizer_state}``. That
        lets ``-load`` restore Adam's m, v moments too, so resumes
        continue from the pre-save trajectory instead of paying a
        ~500-epoch Adam warm-up that can eject a well-converged model
        from its minimum. See ``project_resume_optimizer_state`` memory.

        If ``optimizer`` is None, returns the raw ``state_dict()`` —
        backward-compatible with pre-existing ``.saving`` files.
        """
        if optimizer is None:
            return self.state_dict()
        return {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

    def load(self, saveDict):
        """Load model weights from a checkpoint dict.

        Accepts both formats:
          - Legacy: bare ``state_dict`` produced by ``save()`` without
            ``optimizer`` (all keys are parameter names).
          - New: ``{'model': ..., 'optimizer': ...}`` produced by
            ``save(optimizer=...)``.

        Returns the optimizer state dict if present, else None. Caller
        is responsible for applying it to a freshly-created optimizer
        (Adam's state_dict includes step, m, v; loading restores full
        trajectory).
        """
        if isinstance(saveDict, dict) and 'model' in saveDict:
            self.load_state_dict(saveDict['model'])
            return saveDict.get('optimizer', None)
        # Legacy bare state_dict — no optimizer state available.
        self.load_state_dict(saveDict)
        return None