Tutorial 1: Training Models
===========================

This tutorial covers training the model families in MedLatents:

* Autoregressive Transformer
* MaskGIT
* Discrete Diffusion (D3PM)
* Flow Matching
* Bayesian Flow Networks

There are two layers of API:

* **Self-contained training loops** -- a few lines of standard PyTorch around a
  model's forward pass. These are the copy-pasteable starting points used below
  and in ``examples/quickstart.py``; they run on CPU with synthetic tokens.
* **The full trainer** (:doc:`../api/training`):
  :class:`~medlatents.training.DiscreteLatentTrainer` wraps
  :mod:`accelerate`, EMA, checkpointing, mixed precision, and Weights & Biases
  logging for real workloads. It consumes :class:`~torch.utils.data.DataLoader`
  objects over tokenized medical imagery; see ``scripts/train.py`` for the
  command-line entry point.

Setup
-----

All models share the same size presets and a common set of constructor
arguments (``seq_length``, ``vocab_size``, ``hidden_size``, ``depth``,
``num_heads``):

.. code-block:: python

   import torch
   import torch.nn.functional as F

   from medlatents.configs import MODEL_CONFIGS

   # Problem size and the shared "nano" preset.
   SEQ_LENGTH = 256
   VOCAB_SIZE = 512
   BATCH_SIZE = 32
   NUM_STEPS = 100
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

   config = MODEL_CONFIGS["nano"]   # {'hidden_size': 192, 'depth': 8, 'num_heads': 3}
   print(f"Using device: {DEVICE}")

Create synthetic data
---------------------

For this tutorial we use synthetic token sequences in place of tokenized medical
imagery. In a real workflow these tokens come from a discrete tokenizer (see
:doc:`../token_interface`).

.. code-block:: python

   torch.manual_seed(42)

   train_data = torch.randint(0, VOCAB_SIZE, (1000, SEQ_LENGTH), device=DEVICE)
   val_data = torch.randint(0, VOCAB_SIZE, (100, SEQ_LENGTH), device=DEVICE)

   print(f"Training data shape: {train_data.shape}")
   print(f"Validation data shape: {val_data.shape}")

1. Autoregressive Transformer
-----------------------------

The autoregressive model is trained with a next-token cross-entropy objective:
``model(x)`` returns ``[batch, seq, vocab]`` logits, and each position predicts
the following token.

.. code-block:: python

   from medlatents import AutoregressiveTransformer

   ar_model = AutoregressiveTransformer(
       seq_length=SEQ_LENGTH,
       vocab_size=VOCAB_SIZE,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
       gradient_checkpointing=True,
   ).to(DEVICE)

   optimizer = torch.optim.AdamW(ar_model.parameters(), lr=1e-4)

   ar_model.train()
   for step in range(NUM_STEPS):
       batch = train_data[torch.randint(0, train_data.size(0), (BATCH_SIZE,))]
       optimizer.zero_grad()

       logits = ar_model(batch[:, :-1])          # [B, L-1, vocab]
       loss = F.cross_entropy(
           logits.reshape(-1, logits.size(-1)),
           batch[:, 1:].reshape(-1),
       )
       loss.backward()
       optimizer.step()

   print(f"Final AR training loss: {loss.item():.4f}")

2. MaskGIT
----------

MaskGIT is trained with a masked-token objective. ``model(x)`` returns logits of
shape ``[batch, seq, vocab]``; replace a random subset of tokens with the
model's ``mask_token`` and predict the originals at the masked positions.

.. code-block:: python

   from medlatents import MaskGIT

   maskgit_model = MaskGIT(
       seq_length=SEQ_LENGTH,
       vocab_size=VOCAB_SIZE,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
       gradient_checkpointing=True,
   ).to(DEVICE)

   optimizer = torch.optim.AdamW(maskgit_model.parameters(), lr=1e-4)

   maskgit_model.train()
   for step in range(NUM_STEPS):
       batch = train_data[torch.randint(0, train_data.size(0), (BATCH_SIZE,))]
       optimizer.zero_grad()

       # Randomly mask ~50% of the tokens for this step.
       mask = torch.rand_like(batch, dtype=torch.float) < 0.5
       masked = batch.masked_fill(mask, maskgit_model.mask_token)

       logits = maskgit_model(masked)            # [B, L, vocab]
       loss = F.cross_entropy(
           logits[mask].reshape(-1, logits.size(-1)),
           batch[mask].reshape(-1),
       )
       loss.backward()
       optimizer.step()

   print(f"Final MaskGIT training loss: {loss.item():.4f}")

3. Discrete Diffusion (D3PM)
----------------------------

D3PM trains a denoising network over a discrete corruption process. The
:class:`~medlatents.diffusion.D3PM` object owns the forward process and its
:meth:`~medlatents.diffusion.D3PM.compute_loss` method samples a timestep,
corrupts the batch, and scores the network's prediction -- so the training loop
stays short.

.. code-block:: python

   from medlatents import DiscreteDiT
   from medlatents.diffusion import D3PM

   d3pm_model = DiscreteDiT(
       seq_length=SEQ_LENGTH,
       vocab_size=VOCAB_SIZE,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
   ).to(DEVICE)

   diffusion = D3PM(
       num_classes=VOCAB_SIZE,
       num_timesteps=1000,
       transition_type="absorbing",
       device=DEVICE,
   )

   n_params = sum(p.numel() for p in d3pm_model.parameters()) / 1e6
   print(f"D3PM network with {n_params:.2f}M parameters")

   optimizer = torch.optim.AdamW(d3pm_model.parameters(), lr=1e-4)

   d3pm_model.train()
   for step in range(NUM_STEPS):
       batch = train_data[torch.randint(0, train_data.size(0), (BATCH_SIZE,))]
       optimizer.zero_grad()

       loss = diffusion.compute_loss(d3pm_model, batch, loss_type="cross_entropy")
       loss.backward()
       optimizer.step()

   print(f"Final D3PM training loss: {loss.item():.4f}")

4. Flow Matching
----------------

Flow matching learns a transport map between a source distribution and the data
distribution. Discrete flow matching composes a probability path with a
timestep sampler; the :doc:`../guides/flow_matching` guide and
:class:`~medlatents.flow_matching.DiscreteFlowTrainer` walk through a full path
setup. The backbone is the same :class:`~medlatents.DiscreteDiT` used for D3PM.

.. code-block:: python

   from medlatents import DiscreteDiT
   from medlatents.flow_matching import get_source_distribution

   flow_model = DiscreteDiT(
       seq_length=SEQ_LENGTH,
       vocab_size=VOCAB_SIZE,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
   ).to(DEVICE)

   # A uniform source distribution over the vocabulary.
   source_dist = get_source_distribution("uniform", vocab_size=VOCAB_SIZE)

   print("Flow matching uses a path + trainer; see the flow-matching guide.")

5. Bayesian Flow Networks
-------------------------

BFN frames generation as iterative Bayesian updating of a belief over tokens.
The model takes the same constructor arguments as the other families; the
:doc:`../guides/bayesian_flow` guide covers the accuracy schedules and training
objective in detail.

.. code-block:: python

   from medlatents.bayesian_flow import BayesianFlowTransformer

   bfn_model = BayesianFlowTransformer(
       seq_length=SEQ_LENGTH,
       vocab_size=VOCAB_SIZE,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
   ).to(DEVICE)

   print("BFN training uses an accuracy schedule; see the Bayesian flow guide.")

Scaling up: the full trainer
----------------------------

The loops above are deliberately minimal. For real training runs use
:class:`~medlatents.training.DiscreteLatentTrainer`, which adds
:mod:`accelerate` orchestration, EMA, checkpointing, mixed precision, gradient
accumulation, and Weights & Biases logging. It takes a ``model_type`` string,
:class:`~torch.utils.data.DataLoader` objects, and a configuration object:

.. code-block:: python

   from torch.utils.data import DataLoader, TensorDataset

   from medlatents.training import DiscreteLatentTrainer

   train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
   val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE)

   trainer = DiscreteLatentTrainer(
       model_type="autoreg",       # "autoreg", "maskgit", "d3pm", "flow", "bayesian_flow"
       model=ar_model,
       train_loader=train_loader,
       val_loader=val_loader,
       args=args,                   # run configuration (lr, epochs, output_dir, ...)
   )
   trainer.train()

The command-line entry point ``scripts/train.py`` builds the model, dataloaders,
and ``args`` for you and requires a discrete tokenizer checkpoint
(``--tokenizer_path``) for on-the-fly tokenization of medical imagery.

Saving checkpoints
------------------

After training, save the model weights together with the hyper-parameters needed
to rebuild it. This is the format the generation utilities expect (see
:doc:`02_generation`).

.. code-block:: python

   import os

   os.makedirs("checkpoints", exist_ok=True)

   torch.save(
       {
           "model_state_dict": ar_model.state_dict(),
           "hparams": {
               "seq_length": SEQ_LENGTH,
               "vocab_size": VOCAB_SIZE,
               "hidden_size": config["hidden_size"],
               "depth": config["depth"],
               "num_heads": config["num_heads"],
           },
       },
       "checkpoints/ar_model.pt",
   )

   print("Model saved to checkpoints/")

Training tips
-------------

* **Gradient checkpointing**: enable ``gradient_checkpointing=True`` for long
  sequences to trade compute for memory.
* **Mixed precision**: use :mod:`torch.amp` (or let ``DiscreteLatentTrainer``
  handle it) for faster training on GPU.
* **Learning-rate scheduling**: add warmup and decay for better convergence;
  :func:`~medlatents.training.get_cosine_schedule_with_warmup` is provided.
* **Batch size**: larger batches for stability, smaller for memory constraints.
* **Validation**: validate frequently to detect overfitting early.

Next steps
----------

* :doc:`02_generation` -- generate samples from a trained model.
* :doc:`03_inpainting` -- fill missing regions in images.
* :doc:`04_evaluation` -- compute quality metrics (FID, PSNR, SSIM).
