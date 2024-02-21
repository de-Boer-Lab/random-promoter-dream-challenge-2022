# Prix Fixe Framework for Modular Model Component Testing

The Prix Fixe framework is organized into abstract classes with specific requirements that guide the construction and training of models, Trainers, and DataProcessors.

## Package Structure

The package is structured into several subpackages:

- `prixfixe.prixfixe`: Defines the main abstract classes and sets the requirements for DataProcessor, Network architecture, and Trainer.
- `autosome/bhi/unlockdna`: examples of subpackages demonstrating model reimplementation fitting the framework. They can be used as a reference for developing new models within the Prix Fixe framework.

## Implementation Guide

### DataProcessor

The `DataProcessor` returns an iterable object, such as `DataLoader`, which yields a dictionary containing at least the DNA input `x` and the expression vector `y`. Additional keys can be used by other NN parts. For example, the final layers block of autosome computes loss for the classification task using the `y_probs` key and switch to basic regression only if this key is not present in the batch.

### Trainer

Implement the training procedure within the `Trainer` class. At minimum, define the `train_step` for a single training step and `on_epoch_end` for actions to take at the end of each epoch.

### PrixFixeNet Blocks

The solution must implement three blocks of `PrixFixeNet`:

1. **First Layers Block**: Contains the first layers of the network.
2. **Core Layers Block**: Contains the main architecture components, such as LSTM/self-attention layers.
3. **Final Layers Block**: The final layers that leads to predictions. Also responsible for computing the final loss.

The implementation should include a `train_step` to handle auxiliary losses in the `Trainer`.

The `FinalLayerBlock` must compute the final loss and return it along with the predicted `y`. 