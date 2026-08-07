"""Advanced training losses for generative models.

Provides:
- Consistency distillation losses for faster sampling
- Min-SNR weighting for diffusion training
- Velocity parameterization losses
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


class ConsistencyDistillationLoss(nn.Module):
    """Consistency distillation loss for training fast samplers.

    Trains a student model to match multi-step teacher predictions
    with single-step inference. Key for 1-4 step generation.

    Reference:
    - "Consistency Models" (arXiv:2303.01469)
    - "Improved Techniques for Training Consistency Models" (arXiv:2310.14189)

    Usage:
        loss_fn = ConsistencyDistillationLoss(teacher, student)

        # During training
        x_t, t = sample_noisy_data(x_0)
        loss = loss_fn(student, x_0, x_t, t)
    """

    def __init__(
        self,
        teacher: nn.Module,
        target_ema: nn.Module | None = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_scales: int = 40,
        loss_type: str = "cosine",  # 'l2', 'l1', 'cosine', 'huber'
    ):
        """Initialize consistency distillation loss.

        Args:
            teacher: Pre-trained diffusion teacher model
            target_ema: EMA of student for target (None = use teacher)
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Schedule parameter
            num_scales: Number of discretization steps
            loss_type: Type of reconstruction loss
        """
        super().__init__()
        self.teacher = teacher
        self.target_ema = target_ema
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_scales = num_scales
        self.loss_type = loss_type

        # Precompute sigma schedule
        self.register_buffer(
            "sigmas", self._compute_karras_sigmas(num_scales, sigma_min, sigma_max, rho)
        )

    def _compute_karras_sigmas(
        self,
        n: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
    ) -> torch.Tensor:
        """Compute Karras et al. noise schedule."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def forward(
        self,
        student: nn.Module,
        x_0: torch.Tensor,
        x_t: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute consistency distillation loss.

        Args:
            student: Student model to train
            x_0: Clean data samples [batch, seq_len] or [batch, seq_len, vocab]
            x_t: Noisy samples (computed if None)
            t: Timesteps (sampled if None)

        Returns:
            Scalar loss value
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # Sample timesteps (adjacent pairs)
        n = torch.randint(1, self.num_scales, (batch_size,), device=device)

        # Get sigmas at timesteps (registered buffer)
        sigmas: torch.Tensor = self.get_buffer("sigmas")  # type: ignore[assignment]
        t_n = sigmas[n]
        t_n_minus_1 = sigmas[n - 1]

        # Get noisy samples at t_n
        if x_t is None:
            noise = torch.randn_like(x_0.float())
            x_t_n = x_0.float() + t_n.view(-1, 1) * noise
        else:
            x_t_n = x_t

        # Teacher denoising step: x_t_n -> x_t_{n-1}
        with torch.no_grad():
            # Call teacher model
            if hasattr(self.teacher, "forward"):
                teacher_pred = self.teacher.forward(x_t_n, t_n)
            else:
                teacher_pred = self.teacher(x_t_n, t_n)  # type: ignore
            # Euler step to t_{n-1}
            x_t_n_minus_1 = x_t_n + (t_n_minus_1 - t_n).view(-1, 1) * teacher_pred

        # Student predictions at both timesteps
        student_pred_n = student(x_t_n, t_n)

        # Target: either EMA or teacher at t_{n-1}
        with torch.no_grad():
            if self.target_ema is not None:
                target_pred = self.target_ema(x_t_n_minus_1, t_n_minus_1)
            else:
                target_pred = self.teacher(x_t_n_minus_1, t_n_minus_1)

        # Consistency loss: student at t_n should match target at t_{n-1}
        loss = self._compute_loss(student_pred_n, target_pred)

        return loss

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss between predictions."""
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=0.5)
        elif self.loss_type == "cosine":
            # Cosine distance in feature space (1 - mean cosine similarity).
            pred_norm = F.normalize(pred.float(), dim=-1)
            target_norm = F.normalize(target.float(), dim=-1)
            return 1 - (pred_norm * target_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MinSNRWeightedLoss(nn.Module):
    """Min-SNR weighted loss for balanced diffusion training.

    Weights the loss by min(SNR, gamma) / gamma to balance
    contributions across noise levels.

    Reference:
    - "Efficient Diffusion Training via Min-SNR Weighting Strategy" (arXiv:2303.09556)
    """

    def __init__(
        self,
        gamma: float = 5.0,
        base_loss: str = "mse",
    ):
        """Initialize min-SNR loss.

        Args:
            gamma: SNR clipping threshold (default 5.0)
            base_loss: Base loss function ('mse', 'l1', 'huber')
        """
        super().__init__()
        self.gamma = gamma
        self.base_loss = base_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """Compute min-SNR weighted loss.

        Args:
            pred: Model predictions [batch, ...]
            target: Target values [batch, ...]
            timesteps: Timestep indices [batch]
            alphas_cumprod: Cumulative alpha schedule [num_timesteps]

        Returns:
            Weighted loss
        """
        # Compute SNR at each timestep
        alpha_t = alphas_cumprod[timesteps]
        snr = alpha_t / (1 - alpha_t)

        # Min-SNR weighting
        weights = torch.clamp(snr, max=self.gamma) / self.gamma

        # Compute base loss per sample
        if self.base_loss == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.base_loss == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.base_loss == "huber":
            loss = F.huber_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        # Reduce over non-batch dimensions
        loss = loss.flatten(start_dim=1).mean(dim=1)

        # Apply weights
        weighted_loss = (weights * loss).mean()

        return weighted_loss


class VelocityPredictionLoss(nn.Module):
    """V-prediction loss for improved high-noise training.

    Instead of predicting noise (epsilon), predicts the velocity
    v = alpha * epsilon - sigma * x_0, which has more stable gradients
    at high noise levels.

    Reference:
    - "Progressive Distillation for Fast Sampling of Diffusion Models" (arXiv:2202.00512)
    """

    def __init__(
        self,
        loss_type: str = "mse",
        snr_weighting: bool = True,
        gamma: float = 5.0,
    ):
        """Initialize v-prediction loss.

        Args:
            loss_type: Base loss type
            snr_weighting: Apply SNR weighting
            gamma: SNR clipping threshold
        """
        super().__init__()
        self.loss_type = loss_type
        self.snr_weighting = snr_weighting
        self.gamma = gamma

    def forward(
        self,
        v_pred: torch.Tensor,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute v-prediction loss.

        Args:
            v_pred: Model's velocity prediction [batch, ...]
            x_0: Clean samples [batch, ...]
            noise: Noise that was added [batch, ...]
            alpha: Alpha values at current timesteps [batch, 1, ...]
            sigma: Sigma values at current timesteps [batch, 1, ...]

        Returns:
            Scalar loss
        """
        # Compute target velocity
        v_target = alpha * noise - sigma * x_0

        # Compute base loss
        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Reduce over non-batch dimensions
        loss = loss.flatten(start_dim=1).mean(dim=1)

        # Apply SNR weighting
        if self.snr_weighting:
            snr = (alpha.squeeze() ** 2) / (sigma.squeeze() ** 2 + 1e-8)
            weights = torch.clamp(snr, max=self.gamma) / self.gamma
            loss = weights * loss

        return loss.mean()


class DiscreteFlowMatchingLoss(nn.Module):
    """Flow matching loss for discrete tokens.

    Trains a model to predict the probability flow from source
    to target distribution.

    Reference:
    - "Discrete Flow Matching" (arXiv:2407.15595)
    """

    def __init__(
        self,
        vocab_size: int,
        interpolation: str = "linear",  # 'linear', 'geodesic'
        loss_type: str = "ce",  # 'ce', 'kl'
    ):
        """Initialize discrete flow matching loss.

        Args:
            vocab_size: Size of vocabulary
            interpolation: Type of interpolation between distributions
            loss_type: Loss function type
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.interpolation = interpolation
        self.loss_type = loss_type

    def forward(
        self,
        logits: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discrete flow matching loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            x_1: Target tokens [batch, seq_len]
            x_t: Interpolated tokens at time t [batch, seq_len]
            t: Timesteps [batch]

        Returns:
            Scalar loss
        """
        batch_size, seq_len = x_1.shape

        if self.loss_type == "ce":
            # Cross-entropy: predict target x_1
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), x_1.view(-1), reduction="mean")
        elif self.loss_type == "kl":
            # KL divergence to target one-hot
            probs = F.softmax(logits, dim=-1)
            target_probs = F.one_hot(x_1, self.vocab_size).float()

            # Interpolate target based on t
            t_expanded = t.view(-1, 1, 1)
            target_probs = t_expanded * target_probs + (1 - t_expanded) / self.vocab_size

            # Use safe log to avoid log(0) = -inf
            loss = F.kl_div((probs + 1e-10).log(), target_probs, reduction="batchmean")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Down-weights easy examples and focuses on hard ones.
    Useful when some tokens are much more common than others.

    Reference:
    - "Focal Loss for Dense Object Detection" (arXiv:1708.02002)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """Initialize focal loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights (None for uniform)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [batch, num_classes] or [batch, seq, num_classes]
            targets: Target indices [batch] or [batch, seq]

        Returns:
            Focal loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(targets.device)[targets]
            else:
                alpha_t = self.alpha
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.

    Smooths one-hot targets by mixing with uniform distribution.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label-smoothed cross entropy."""
        vocab_size = logits.size(-1)

        # Flatten if needed
        if logits.dim() == 3:
            batch, seq_len, _ = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
        else:
            batch, seq_len = logits.size(0), 1

        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / vocab_size)
        smooth_targets.scatter_(
            1, targets.unsqueeze(1), 1 - self.smoothing + self.smoothing / vocab_size
        )

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.view(batch, seq_len)


class AdaptiveTemperatureScaling(nn.Module):
    """Learnable temperature scaling for logits."""

    def __init__(
        self,
        mode: str = "global",
        init_temp: float = 1.0,
        seq_length: int | None = None,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.mode = mode

        if mode == "global":
            self.temperature = nn.Parameter(torch.tensor(init_temp))
        elif mode == "per_position":
            if seq_length is None:
                raise ValueError("seq_length required for per_position mode")
            self.temperature = nn.Parameter(torch.full((seq_length,), init_temp))
        elif mode == "per_class":
            if vocab_size is None:
                raise ValueError("vocab_size required for per_class mode")
            self.temperature = nn.Parameter(torch.full((vocab_size,), init_temp))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "global":
            return logits / self.temperature
        elif self.mode == "per_position":
            return logits / self.temperature.view(1, -1, 1)
        elif self.mode == "per_class":
            return logits / self.temperature.view(1, 1, -1)
        return logits


class TokenDropout(nn.Module):
    """Token-level dropout for regularization."""

    def __init__(
        self,
        p: float = 0.1,
        mask_token: int | None = None,
        vocab_size: int | None = None,
        mode: str = "mask",  # 'mask' or 'random'
    ):
        super().__init__()
        self.p = p
        self.mask_token = mask_token
        self.vocab_size = vocab_size
        self.mode = mode

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return tokens

        mask = torch.rand_like(tokens.float()) < self.p
        output = tokens.clone()

        if self.mode == "mask":
            if self.mask_token is None:
                raise ValueError("mask_token required for mask mode")
            output[mask] = self.mask_token
        elif self.mode == "random":
            if self.vocab_size is None:
                raise ValueError("vocab_size required for random mode")
            random_tokens = torch.randint(0, self.vocab_size, tokens.shape, device=tokens.device)
            output[mask] = random_tokens[mask]

        return output


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss."""
        # Soft loss (KL divergence between soft predictions)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            self.temperature**2
        )

        # Hard loss (cross entropy with true labels)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)), targets.view(-1)
        )

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute perplexity from logits and targets."""
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    return torch.exp(loss)


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token-level accuracy."""
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0)
    correct = (preds == targets) & mask
    return correct.sum().float() / mask.sum().float()


class VelocityContrastiveRegularization(nn.Module):
    """VeCoR: Velocity Contrastive Regularization for Flow Matching.

    Extends standard flow matching with two-sided supervision:
    - Positive: Push predicted velocity toward correct target (standard FM)
    - Negative: Push predicted velocity away from incorrect targets

    This contrastive formulation regularizes trajectory evolution and improves
    perceptual fidelity, especially in low-step and lightweight configurations.

    Reference:
        Hong et al., "VeCoR - Velocity Contrastive Regularization for Flow Matching"
        arXiv:2511.18942

    Results:
        - 22% FID reduction on SiT-XL/2
        - 35% FID reduction on REPA-SiT-XL/2
        - 32% FID reduction on text-to-image (MS-COCO)

    Args:
        temperature: Temperature for contrastive softmax (lower = harder negatives)
        weight: Weight for VeCoR loss term relative to main loss
        negative_mode: How to construct negative samples
            - 'roll': Roll batch dimension (default, simple and effective)
            - 'shuffle': Random permutation
            - 'hard': Use semantically hard negatives (requires additional info)
        margin: Optional margin for contrastive loss (0 = no margin)

    Usage:
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1)

        # In training loop:
        v_pred = model(x_t, t)
        v_target = x_1 - x_0  # True velocity

        fm_loss = F.mse_loss(v_pred, v_target)
        vecor_loss = vecor(v_pred, v_target)

        total_loss = fm_loss + vecor_loss
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 0.1,
        negative_mode: str = "roll",
        margin: float = 0.0,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        if negative_mode not in ("roll", "shuffle"):
            raise ValueError(f"negative_mode must be 'roll' or 'shuffle', got {negative_mode}")

        self.temperature = temperature
        self.weight = weight
        self.negative_mode = negative_mode
        self.margin = margin

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VeCoR contrastive loss.

        Args:
            v_pred: Predicted velocity [batch, seq_len, dim] or [batch, C, H, W]
            v_target: Target velocity, same shape as v_pred

        Returns:
            Weighted contrastive loss (scalar)
        """
        batch_size = v_pred.shape[0]
        if batch_size < 2:
            # Need at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)

        # Create negative targets by rolling/shuffling batch
        if self.negative_mode == "roll":
            v_negative = torch.roll(v_target, shifts=1, dims=0)
        else:  # shuffle
            perm = torch.randperm(batch_size, device=v_target.device)
            # Ensure no sample is paired with itself
            while (perm == torch.arange(batch_size, device=v_target.device)).any():
                perm = torch.randperm(batch_size, device=v_target.device)
            v_negative = v_target[perm]

        # Flatten spatial/sequence dimensions for similarity computation
        v_pred_flat = v_pred.flatten(start_dim=1)  # [batch, dim]
        v_target_flat = v_target.flatten(start_dim=1)
        v_negative_flat = v_negative.flatten(start_dim=1)

        # L2 normalize for cosine similarity
        v_pred_norm = F.normalize(v_pred_flat, dim=-1, p=2)
        v_target_norm = F.normalize(v_target_flat, dim=-1, p=2)
        v_negative_norm = F.normalize(v_negative_flat, dim=-1, p=2)

        # Compute similarities (dot product of normalized vectors = cosine similarity)
        pos_sim = (v_pred_norm * v_target_norm).sum(dim=-1)  # [batch]
        neg_sim = (v_pred_norm * v_negative_norm).sum(dim=-1)  # [batch]

        # Apply margin if specified
        if self.margin > 0:
            pos_sim = pos_sim - self.margin

        # Scale by temperature
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature

        # InfoNCE-style contrastive loss
        # Loss = -log(exp(pos) / (exp(pos) + exp(neg)))
        # Rewritten using log-sum-exp trick for numerical stability:
        # = -pos + log(exp(pos) + exp(neg)) = -pos + logsumexp([pos, neg])
        # = logsumexp([pos, neg]) - pos
        stacked = torch.stack([pos_sim, neg_sim], dim=-1)
        loss = torch.logsumexp(stacked, dim=-1) - pos_sim

        return self.weight * loss.mean()


class ContrastiveFlowMatchingLoss(nn.Module):
    """Contrastive Flow Matching Loss combining standard FM with VeCoR.

    A complete loss function for flow matching that includes:
    1. Standard MSE velocity prediction loss
    2. VeCoR contrastive regularization

    This is a convenience wrapper that combines both losses with proper weighting.

    Args:
        vecor_temperature: Temperature for VeCoR contrastive loss
        vecor_weight: Weight for VeCoR term (relative to MSE loss)
        reduction: Reduction mode for MSE loss ('mean', 'sum', 'none')

    Usage:
        loss_fn = ContrastiveFlowMatchingLoss(vecor_weight=0.1)

        # In training:
        v_pred = model(x_t, t)
        v_target = x_1 - x_0

        loss = loss_fn(v_pred, v_target)
    """

    def __init__(
        self,
        vecor_temperature: float = 0.1,
        vecor_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.vecor = VelocityContrastiveRegularization(
            temperature=vecor_temperature,
            weight=vecor_weight,
        )
        self.reduction = reduction

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined flow matching loss.

        Args:
            v_pred: Predicted velocity
            v_target: Target velocity

        Returns:
            Tuple of (total_loss, mse_loss, vecor_loss)
        """
        # Standard MSE loss
        mse_loss = F.mse_loss(v_pred, v_target, reduction=self.reduction)

        # VeCoR contrastive loss
        vecor_loss = self.vecor(v_pred, v_target)

        total_loss = mse_loss + vecor_loss

        return total_loss, mse_loss, vecor_loss


__all__ = [
    "ConsistencyDistillationLoss",
    "MinSNRWeightedLoss",
    "VelocityPredictionLoss",
    "DiscreteFlowMatchingLoss",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "AdaptiveTemperatureScaling",
    "TokenDropout",
    "DistillationLoss",
    "VelocityContrastiveRegularization",
    "ContrastiveFlowMatchingLoss",
    "compute_perplexity",
    "compute_token_accuracy",
]
