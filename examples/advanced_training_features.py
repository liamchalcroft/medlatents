"""Advanced training features demonstration.

This example demonstrates the cutting-edge diffusion training techniques
implemented in medlatents:

1. ConditioningBundle - Unified conditioning interface for multi-modal inputs
2. REPA - Representation Alignment for 17x faster training (ICLR'25 Oral)
3. HASTE - Optimal termination of alignment loss (28x speedup)
4. VeCoR - Velocity Contrastive Regularization (22-35% FID reduction)
5. REG - Rectified Gradient Guidance (ICML'25)
6. REPA-E - End-to-end VAE + DiT training (45x speedup)
7. SPRINT - Token dropping for efficient training (9.8x savings)
8. Cross-attention DiT - Text/image conditioning
9. RMSNorm & Value Residual - Modern architectural improvements

Usage:
    # Demo mode (synthetic data)
    python examples/advanced_training_features.py --demo

    # Full training with REPA
    python examples/advanced_training_features.py \
        --data_path /path/to/data \
        --use_repa --use_haste

    # Training with SPRINT token dropping
    python examples/advanced_training_features.py \
        --data_path /path/to/data \
        --use_sprint --drop_ratio 0.5

    # Inference with REG guidance
    python examples/advanced_training_features.py \
        --checkpoint /path/to/checkpoint.pt \
        --use_reg --guidance_scale 2.0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Core imports from medlatents
# ============================================================================
from medlatents.conditioning import (
    ConditioningBundle,
    ConditioningConfig,
)
from medlatents.networks import (
    ContinuousDiT,
    SPRINTConfig,
    SPRINTDiT,
)
from medlatents.sampling.flow_diffusion import (
    REGConfig,
    REGFlowMatcher,
    sample_with_cfg_flow,
    sample_with_reg_flow,
)
from medlatents.training import (
    ConditionalTrainingConfig,
    ContrastiveFlowMatchingLoss,
    HASTEScheduler,
    # REPA-E
    REPAEConfig,
    REPAETrainer,
    REPALoss,
    # REPA alignment
    REPAProjection,
    # VeCoR contrastive loss
    VelocityContrastiveRegularization,
    prepare_batch_with_conditioning,
)

# ============================================================================
# Demo: ConditioningBundle usage
# ============================================================================


def demo_conditioning_bundle():
    """Demonstrate ConditioningBundle for multi-modal conditioning."""
    print("\n" + "=" * 60)
    print("Demo: ConditioningBundle - Unified Conditioning Interface")
    print("=" * 60)

    # Define what conditioning the model supports
    config = ConditioningConfig(
        use_timestep=True,
        use_class=True,
        num_classes=10,
        class_dropout_prob=0.1,  # 10% CFG dropout
        use_text=True,
        text_embed_dim=768,
        use_spatial=True,
        spatial_channels=4,
    )
    config.validate()
    print(f"\nConditioning config: {config}")

    # Create a conditioning bundle
    batch_size = 4
    bundle = ConditioningBundle(
        timesteps=torch.rand(batch_size),
        class_labels=torch.randint(0, 10, (batch_size,)),
        text_embeddings=torch.randn(batch_size, 77, 768),
        text_pooled=torch.randn(batch_size, 768),
        spatial_condition=torch.randn(batch_size, 64, 4),  # e.g., inpainting mask
    )
    print("\nOriginal bundle:")
    print(f"  - timesteps: {bundle.timesteps.shape}")
    print(f"  - class_labels: {bundle.class_labels}")
    print(f"  - text_embeddings: {bundle.text_embeddings.shape}")
    print(f"  - spatial_condition: {bundle.spatial_condition.shape}")

    # Apply CFG dropout for training
    bundle_dropped = bundle.apply_cfg_dropout(
        class_dropout_prob=0.1,
        text_dropout_prob=0.1,
    )
    print("\nAfter CFG dropout:")
    print(f"  - class_labels (may be replaced with null): {bundle_dropped.class_labels}")

    # Get null bundle for inference
    null_bundle = bundle.get_null_bundle()
    print("\nNull bundle for CFG inference:")
    print(f"  - class_labels (all null): {null_bundle.class_labels}")
    print(f"  - text_embeddings (zeros): {null_bundle.text_embeddings.sum().item():.4f}")

    # Create bundle from dataloader batch
    batch_dict = {
        "data": torch.randn(batch_size, 16, 32),
        "class_labels": torch.randint(0, 10, (batch_size,)),
        "text_embeddings": torch.randn(batch_size, 77, 768),
    }
    timesteps = torch.rand(batch_size)
    bundle_from_batch = ConditioningBundle.from_batch(batch_dict, timesteps, config)
    print(f"\nBundle from batch dict: class_labels={bundle_from_batch.class_labels}")


# ============================================================================
# Demo: REPA (Representation Alignment)
# ============================================================================


def demo_repa_training():
    """Demonstrate REPA alignment loss for faster training."""
    print("\n" + "=" * 60)
    print("Demo: REPA - Representation Alignment (17x Speedup)")
    print("=" * 60)

    # Create REPA projection head
    hidden_dim = 256  # DiT hidden dimension
    target_dim = 768  # DINOv2-B dimension
    proj_dim = 128  # Projection dimension

    projection = REPAProjection(
        hidden_dim=hidden_dim,
        target_dim=target_dim,
        proj_dim=proj_dim,
        num_layers=2,
    )
    print(f"\nREPA Projection: {hidden_dim} -> {proj_dim} (target: {target_dim})")

    # Create REPA loss
    repa_loss = REPALoss(
        hidden_dim=hidden_dim,
        target_dim=target_dim,
        proj_dim=proj_dim,
        timestep_threshold=0.5,  # Only apply at t > 0.5 (high noise)
        weight=0.5,
    )
    print(f"REPA Loss: threshold={repa_loss.timestep_threshold}, weight={repa_loss.weight}")

    # Simulate training step
    batch_size, seq_len = 4, 64
    dit_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_features = torch.randn(batch_size, seq_len, target_dim)
    timesteps = torch.rand(batch_size)

    # Compute REPA loss directly (it handles projection internally)
    loss = repa_loss(dit_hidden, encoder_features, timesteps=timesteps)
    print(f"\nREPA loss: {loss.item():.4f}")

    # Or manually project and compute similarity
    proj_hidden = projection(dit_hidden)
    proj_target = projection.project_target(encoder_features)
    print(f"Projected hidden: {proj_hidden.shape}, target: {proj_target.shape}")

    # HASTE: Schedule alignment termination
    haste = HASTEScheduler(
        termination_step=10000,
        termination_mode="linear",
        transition_steps=1000,
    )
    print("\nHASTE REPA weights at different steps:")
    for step in [0, 5000, 9000, 10000, 11000]:
        weight = haste.get_repa_weight(step)
        print(f"  Step {step}: weight = {weight:.3f}")


# ============================================================================
# Demo: VeCoR (Velocity Contrastive Regularization)
# ============================================================================


def demo_vecor_loss():
    """Demonstrate VeCoR for improved flow matching."""
    print("\n" + "=" * 60)
    print("Demo: VeCoR - Velocity Contrastive Regularization")
    print("=" * 60)

    # Create VeCoR loss
    vecor = VelocityContrastiveRegularization(
        temperature=0.1,
        weight=0.1,
    )
    print(f"\nVeCoR: temperature={vecor.temperature}, weight={vecor.weight}")

    # Simulate predictions
    batch_size, seq_len, dim = 4, 64, 32
    pred_velocity = torch.randn(batch_size, seq_len, dim)
    target_velocity = torch.randn(batch_size, seq_len, dim)

    loss = vecor(pred_velocity, target_velocity)
    print(f"VeCoR loss: {loss.item():.4f}")

    # Combined flow matching loss with VeCoR
    cfm_loss = ContrastiveFlowMatchingLoss(
        vecor_weight=0.1,
        vecor_temperature=0.1,
    )
    total_loss, mse_loss, vecor_loss_part = cfm_loss(pred_velocity, target_velocity)
    print(
        f"Combined CFM+VeCoR loss: {total_loss.item():.4f} (MSE: {mse_loss.item():.4f}, VeCoR: {vecor_loss_part.item():.4f})"
    )


# ============================================================================
# Demo: REG (Rectified Gradient Guidance)
# ============================================================================


def demo_reg_guidance():
    """Demonstrate REG for improved CFG."""
    print("\n" + "=" * 60)
    print("Demo: REG - Rectified Gradient Guidance (ICML'25)")
    print("=" * 60)

    # Create a simple flow model for demo
    class SimpleFlowModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),  # +1 for time
                nn.SiLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x_t, time, condition=None):
            t = time.view(-1, 1).expand(-1, x_t.shape[1]).unsqueeze(-1)
            x_with_t = torch.cat([x_t, t], dim=-1)
            return self.net(x_with_t)

    model = SimpleFlowModel(dim=32)
    print("\nSimple flow model created")

    # REG configuration
    reg_config = REGConfig(
        guidance_scale=1.5,
        noise_schedule="flow",
        schedule_type="constant",
    )
    print(f"REG config: scale={reg_config.guidance_scale}, schedule={reg_config.schedule_type}")

    # Sample with REG
    batch_size, seq_len, dim = 2, 16, 32
    x_t = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    t = torch.tensor([0.5, 0.7])
    condition = torch.randn(batch_size, 64)  # Some conditioning

    # Standard CFG
    cfg_output = sample_with_cfg_flow(
        model,
        x_t.detach(),
        t,
        condition,
        guidance_scale=1.5,
        null_condition=torch.zeros_like(condition),
    )
    print(f"\nStandard CFG output: {cfg_output.shape}")

    # REG guidance
    x_t_reg = x_t.detach().requires_grad_(True)
    reg_output = sample_with_reg_flow(
        model,
        x_t_reg,
        t,
        condition,
        reg_config=reg_config,
        null_condition=torch.zeros_like(condition),
    )
    print(f"REG output: {reg_output.shape}")

    # REG wrapper for easy integration
    reg_wrapper = REGFlowMatcher(model, reg_config)
    wrapped_output = reg_wrapper(x_t.detach().requires_grad_(True), t, condition)
    print(f"REG wrapper output: {wrapped_output.shape}")


# ============================================================================
# Demo: SPRINT (Token Dropping)
# ============================================================================


def demo_sprint():
    """Demonstrate SPRINT token dropping for efficient training."""
    print("\n" + "=" * 60)
    print("Demo: SPRINT - Sparse-Dense Residual Fusion (9.8x Savings)")
    print("=" * 60)

    # Create base DiT
    base_dit = ContinuousDiT(
        seq_length=64,
        in_channels=32,
        hidden_size=256,
        depth=12,
        num_heads=4,
    )
    print(f"\nBase DiT: {sum(p.numel() for p in base_dit.parameters())} parameters")

    # Wrap with SPRINT
    sprint_config = SPRINTConfig(
        drop_ratio=0.5,  # Drop 50% of tokens
        dense_layers=6,  # First 6 layers process all tokens
        sparse_layers=6,  # Last 6 layers process sparse tokens
        fusion_method="residual",
        importance_type="random",
    )
    sprint_dit = SPRINTDiT(base_dit, sprint_config)
    print(
        f"SPRINT config: drop_ratio={sprint_config.drop_ratio}, "
        f"dense={sprint_config.dense_layers}, sparse={sprint_config.sparse_layers}"
    )

    # Forward pass comparison
    batch_size, seq_len, channels = 4, 64, 32
    x = torch.randn(batch_size, seq_len, channels)
    t = torch.rand(batch_size)

    # Full forward (inference)
    sprint_dit.eval()
    with torch.no_grad():
        full_output = sprint_dit(x, t, training=False)
    print(f"\nFull forward output: {full_output.shape}")

    # Sparse forward (training)
    sprint_dit.train()
    sparse_output = sprint_dit(x, t, training=True)
    print(f"Sparse forward output: {sparse_output.shape}")

    # Token selection demo
    from medlatents.networks.sprint import TokenRestorer, TokenSelector

    selector = TokenSelector(hidden_size=256, importance_type="random")
    hidden = torch.randn(batch_size, seq_len, 256)

    selected, indices, mask = selector(hidden, keep_ratio=0.5)
    print(f"\nToken selection: {hidden.shape} -> {selected.shape}")
    print(f"Kept indices shape: {indices.shape}")
    print(f"Mask sum: {mask.sum(dim=1)}  (should be ~{seq_len * 0.5})")

    # Restoration
    restorer = TokenRestorer(hidden_size=256, fusion_method="residual")
    restored = restorer(selected, hidden, indices, mask)
    print(f"Restored shape: {restored.shape}")


# ============================================================================
# Demo: REPA-E (End-to-End VAE Training)
# ============================================================================


def demo_repa_e():
    """Demonstrate REPA-E for end-to-end VAE + DiT training."""
    print("\n" + "=" * 60)
    print("Demo: REPA-E - End-to-End VAE + DiT Training (45x Speedup)")
    print("=" * 60)

    # Create simple VAE
    class SimpleVAE(nn.Module):
        def __init__(self, in_channels=3, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, latent_dim * 2, 4, 2, 1),  # mu + logvar
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            )

        def encode(self, x):
            h = self.encoder(x)
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar

        def decode(self, z):
            return self.decoder(z)

    # Create simple frozen encoder
    class SimpleEncoder(nn.Module):
        def __init__(self, out_dim=768):
            super().__init__()
            self.proj = nn.Linear(32 * 8 * 8, out_dim)

        def forward(self, x):
            # Flatten and project
            return self.proj(x.flatten(1)).unsqueeze(1).expand(-1, 64, -1)

    vae = SimpleVAE(in_channels=3, latent_dim=32)
    frozen_encoder = SimpleEncoder(out_dim=768)
    frozen_encoder.eval()

    # Create DiT
    dit = ContinuousDiT(
        seq_length=64,
        in_channels=32,
        hidden_size=256,
        depth=6,
        num_heads=4,
    )

    # Create REPA projection
    projection = REPAProjection(
        hidden_dim=256,
        target_dim=768,
        proj_dim=128,
    )

    # REPA-E config
    config = REPAEConfig(
        repa_weight=1.0,
        diffusion_weight=1.0,
        timestep_threshold=0.5,
        vae_grad_scale=0.1,  # Lower LR for VAE
    )
    print(
        f"\nREPA-E config: repa_weight={config.repa_weight}, "
        f"diffusion_weight={config.diffusion_weight}"
    )

    # Create trainer
    trainer = REPAETrainer(
        dit=dit,
        vae=vae,
        frozen_encoder=frozen_encoder,
        repa_projection=projection,
        config=config,
    )
    print("REPA-E trainer created")

    # Simulate training step
    images = torch.randn(4, 3, 32, 32)  # Batch of images
    timesteps = torch.rand(4)

    # Note: In real usage, you'd have proper data and longer training
    # This is just to demonstrate the API
    print(f"\nInput images: {images.shape}")
    print(f"Timesteps: {timesteps.shape}")

    # Get optimizer groups (separate LR for VAE)
    groups = trainer.create_optimizer_groups(
        dit_lr=1e-4,
        vae_lr=1e-5,  # Lower LR for VAE
    )
    print("\nOptimizer groups:")
    print(f"  - DiT: {len(groups[0]['params'])} param tensors, lr={groups[0]['lr']}")
    print(f"  - VAE: {len(groups[1]['params'])} param tensors, lr={groups[1]['lr']}")


# ============================================================================
# Demo: Cross-attention DiT
# ============================================================================


def demo_cross_attention_dit():
    """Demonstrate cross-attention DiT for text/image conditioning."""
    print("\n" + "=" * 60)
    print("Demo: Cross-Attention DiT - Multi-Modal Conditioning")
    print("=" * 60)

    from medlatents.networks.diffusion_transformer import (
        DiTBlockWithCrossAttention,
        MMDiTBlock,
    )
    from medlatents.networks.transformer import precompute_freqs_cis

    hidden_size = 256
    context_dim = 768  # Text embedding dimension
    num_heads = 4
    head_dim = hidden_size // num_heads

    # Create blocks
    cross_attn_block = DiTBlockWithCrossAttention(
        hidden_size=hidden_size,
        context_dim=context_dim,
        num_heads=num_heads,
    )
    mm_block = MMDiTBlock(
        hidden_size=hidden_size,
        context_dim=context_dim,
        num_heads=num_heads,
    )
    print(f"\nDiTBlockWithCrossAttention: hidden={hidden_size}, context={context_dim}")
    print(f"MMDiTBlock (SD3-style): hidden={hidden_size}, context={context_dim}")

    # Create inputs
    batch_size, seq_len = 4, 64
    context_len = 77  # Text sequence length

    x = torch.randn(batch_size, seq_len, hidden_size)
    c = torch.randn(batch_size, hidden_size)  # Timestep + class embedding
    context = torch.randn(batch_size, context_len, context_dim)  # Text embeddings
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    # Forward with cross-attention
    out_cross = cross_attn_block(x, c, freqs_cis, context=context)
    print(f"\nCross-attention block output: {out_cross.shape}")

    # Forward with MM-DiT (joint attention)
    out_mm = mm_block(x, c, freqs_cis, context=context)
    print(f"MM-DiT block output: {out_mm.shape}")

    # Without context (falls back to self-attention only)
    out_no_ctx = cross_attn_block(x, c, freqs_cis, context=None)
    print(f"Without context: {out_no_ctx.shape}")


# ============================================================================
# Demo: Putting it all together
# ============================================================================


def demo_full_pipeline():
    """Demonstrate a complete training pipeline with all features."""
    print("\n" + "=" * 60)
    print("Demo: Complete Training Pipeline")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 64
    hidden_size = 256
    in_channels = 32

    # 1. Create model with modern architecture
    dit = ContinuousDiT(
        seq_length=seq_len,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=12,
        num_heads=4,
        num_classes=10,
        class_dropout_prob=0.1,
        qk_norm=True,  # QK LayerNorm for stability
    )
    print("\n1. Created DiT with QK-norm")

    # 2. Optionally wrap with SPRINT
    sprint_config = SPRINTConfig(drop_ratio=0.5, dense_layers=6, sparse_layers=6)
    sprint_dit = SPRINTDiT(dit, sprint_config)
    print("2. Wrapped with SPRINT (50% token dropping)")

    # 3. Setup conditioning
    cond_config = ConditioningConfig(
        use_class=True,
        num_classes=10,
        class_dropout_prob=0.1,
    )
    print("3. Configured conditioning (10 classes, 10% CFG dropout)")

    # 4. Setup REPA
    _repa_proj = REPAProjection(hidden_size, target_dim=768, proj_dim=128)
    _repa_loss = REPALoss(hidden_size, target_dim=768, proj_dim=128)
    _haste = HASTEScheduler(termination_step=10000)
    print("4. Setup REPA + HASTE")
    del _repa_proj, _repa_loss, _haste

    # 5. Setup VeCoR
    vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1)
    print("5. Setup VeCoR contrastive loss")

    # 6. Create training config
    train_config = ConditionalTrainingConfig(
        conditioning_config=cond_config,
        use_repa=True,
        use_vecor=True,
        use_haste=True,
        haste_termination_step=10000,
    )
    print("6. Created training config")

    # 7. Simulate one training step
    print("\n7. Simulating training step...")

    # Create synthetic batch
    batch = {
        "data": torch.randn(batch_size, seq_len, in_channels),
        "class_labels": torch.randint(0, 10, (batch_size,)),
    }
    timesteps = torch.rand(batch_size)

    # Prepare batch with conditioning
    data, bundle = prepare_batch_with_conditioning(batch, timesteps, train_config, training=True)
    print("   - Prepared batch with conditioning")
    print(f"   - Data: {data.shape}, Class labels: {bundle.class_labels}")

    # Forward pass (simplified - real code would add noise first)
    sprint_dit.train()
    output = sprint_dit(data, timesteps, y=bundle.class_labels, training=True)
    print(f"   - Model output: {output.shape}")

    # Compute loss (simplified)
    target = torch.randn_like(output)
    mse_loss = F.mse_loss(output, target)
    vecor_loss = vecor(output, target)
    total_loss = mse_loss + vecor_loss
    print(f"   - MSE loss: {mse_loss.item():.4f}")
    print(f"   - VeCoR loss: {vecor_loss.item():.4f}")
    print(f"   - Total loss: {total_loss.item():.4f}")

    # 8. REG for inference
    print("\n8. Setup REG for inference...")
    reg_config = REGConfig(guidance_scale=2.0, noise_schedule="flow")
    print(f"   - REG scale: {reg_config.guidance_scale}")

    print("\n" + "=" * 60)
    print("Pipeline demo complete!")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Advanced Training Features Demo")
    parser.add_argument("--demo", action="store_true", help="Run all demos")
    parser.add_argument("--conditioning", action="store_true", help="Demo ConditioningBundle")
    parser.add_argument("--repa", action="store_true", help="Demo REPA")
    parser.add_argument("--vecor", action="store_true", help="Demo VeCoR")
    parser.add_argument("--reg", action="store_true", help="Demo REG")
    parser.add_argument("--sprint", action="store_true", help="Demo SPRINT")
    parser.add_argument("--repa_e", action="store_true", help="Demo REPA-E")
    parser.add_argument("--cross_attn", action="store_true", help="Demo cross-attention")
    parser.add_argument("--pipeline", action="store_true", help="Demo full pipeline")
    args = parser.parse_args()

    # Run all if --demo or no specific flag
    run_all = args.demo or not any(
        [
            args.conditioning,
            args.repa,
            args.vecor,
            args.reg,
            args.sprint,
            args.repa_e,
            args.cross_attn,
            args.pipeline,
        ]
    )

    print("=" * 60)
    print("medlatents Advanced Training Features Demo")
    print("=" * 60)

    if run_all or args.conditioning:
        demo_conditioning_bundle()

    if run_all or args.repa:
        demo_repa_training()

    if run_all or args.vecor:
        demo_vecor_loss()

    if run_all or args.reg:
        demo_reg_guidance()

    if run_all or args.sprint:
        demo_sprint()

    if run_all or args.repa_e:
        demo_repa_e()

    if run_all or args.cross_attn:
        demo_cross_attention_dit()

    if run_all or args.pipeline:
        demo_full_pipeline()


if __name__ == "__main__":
    main()
