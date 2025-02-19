import torch
import numpy as np

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    max_start_index = len(x) - context_length
    if max_start_index <= 0:
        raise ValueError("Dataset too small for given context length.")

    # Randomly sample batch_size starting indices
    start_indices = np.random.randint(0, max_start_index, size=batch_size)

    # np.memmap
    if isinstance(x, np.memmap):
        inputs = np.empty((batch_size, context_length), dtype=x.dtype)
        targets = np.empty((batch_size, context_length), dtype=x.dtype)

        for i, idx in enumerate(start_indices):
            inputs[i] = x[idx : idx + context_length]
            targets[i] = x[idx + 1 : idx + context_length + 1]
    else:
        inputs = np.stack([x[i : i + context_length] for i in start_indices])
        targets = np.stack([x[i + 1 : i + context_length + 1] for i in start_indices])

    #Change input to np.int64 because uint16 is not supported
    if inputs.dtype == np.uint16:
        inputs = inputs.astype(np.int64, copy=False)
    if targets.dtype == np.uint16:
        targets = targets.astype(np.int64, copy=False)

    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor
