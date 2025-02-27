import os
import torch
import torch.nn as nn
import math
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.versions import require_version
from peft import get_peft_model, LoraConfig, TaskType

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def fft_svd_projection(A, k):
    """
    Use FFT to implement low-rank approximation projection.
    """
    m, n = A.shape
    A_fft = torch.fft.fft2(A)
    A_fft_k = torch.zeros_like(A_fft)
    A_fft_k[:, :k] = A_fft[:, :k]
    A_k = torch.fft.ifft2(A_fft_k).real
    return A_k[:, :k]

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device), full_rank_grad)
        # Extend other proj_type ...
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device), low_rank_grad)
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights
        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        A_proj = fft_svd_projection(matrix, k=rank)
        if type == 'left':
            result = A_proj
        elif type == 'right':
            A_proj_right = fft_svd_projection(matrix.T, k=rank)
            result = A_proj_right.T
        else:
            raise ValueError("type should be 'left' or 'right'")
        if not float_data:
            result = result.to(original_device).type(original_type)
        return result

# Another GaLore projector (Tensor version, optional)
class GaLoreProjectorTensor:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank_all):
        # Use tensorly for Tucker decomposition (requires tensorly library)
        from tensorly.decomposition import tucker
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all)
        return tucker_tensor

    def transform(self, tensor, x):
        from tensorly import tenalg
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        from tensorly import tenalg
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)

# Custom optimizer
class FFT_GaLoreAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn("This implementation of AdamW is deprecated, consider using torch.optim.AdamW", FutureWarning)
        require_version("torch>=1.5.0")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                # GaLore Projection (if the group contains rank parameter)
                if "rank" in group:
                    if "projector" not in state:
                        # Choose the appropriate projector based on parameter dimensions
                        if grad.dim() <= 2:
                            from copy import deepcopy
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"])
                    grad = state["projector"].project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                norm_grad = exp_avg / denom
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                p.add_(norm_grad, alpha=-step_size)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        return loss
