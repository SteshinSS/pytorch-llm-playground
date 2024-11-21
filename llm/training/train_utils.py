import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


def calc_loss_batch(input_batch, target_batch, model, device) -> Float[Tensor, ""]:
    input_batch = input_batch.to(device)
    target_batch: Int[Tensor, "batch seq"] = target_batch.to(device)
    target_batch: Int[Tensor, " batchXseq"] = rearrange(target_batch, "b s -> (b s)")
    logits: Float[Tensor, "batch seq vocab_size"] = model(input_batch)
    logits: Float[Tensor, "batchXseq vocab_size"] = rearrange(logits, "b s v -> (b s) v")
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches
