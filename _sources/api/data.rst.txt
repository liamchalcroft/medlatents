Data API
========

Datasets, dataloaders, and tokenizer wrappers for feeding tokenized medical
imagery into the models. Discrete models consume integer codebook indices;
continuous models consume floating-point latents. See :doc:`../token_interface`
for the exact token schema, shapes, and validation rules.

Datasets
--------

.. autoclass:: medlatents.data.TokenizedImageDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.data.ContinuousLatentDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.data.PreTokenizedDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.data.PairedDataset
   :members:
   :undoc-members:
   :show-inheritance:

Dataloaders
-----------

.. autofunction:: medlatents.data.get_tokenized_dataloaders

.. autofunction:: medlatents.data.get_latent_dataloaders

.. autofunction:: medlatents.data.pretokenize_dataset

Tokenizers
----------

Thin wrappers over ``medtokenizers`` discrete and continuous tokenizers.

.. autoclass:: medlatents.data.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.data.DiscreteTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.data.ContinuousTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.data.get_tokenizer_info
