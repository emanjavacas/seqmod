
import torch


def scheduled_sampling(inp, prev_out, project, exposure_rate):
    """
    Resample n inputs to next iteration from the model itself. N is itself
    sampled from a bernoulli independently for each example in the batch
    with weights equal to the model's variable self.scheduled_rate.

    Parameters:
    -----------

    - inp: torch.tensor(batch_size, dtype=int)
    - dec_out: torch.Tensor(batch_size x hid_dim)
    - project: nn.Module to compute the model output from hidden
    - exposure_rate: float, (0.0, 1.0)

    Returns: partially resampled input
    --------
    - inp: torch.tensor(batch_size, dtype=int)
    """
    with torch.no_grad():

        keep_mask = torch.bernoulli(
            torch.zeros_like(inp).type(torch.float) + exposure_rate) == 1

        # return if no sampling is necessary
        if len(keep_mask.nonzero()) == len(inp):
            return inp

        sampled = project(prev_out).max(1)[1]

        if len(keep_mask.nonzero()) == 0:  # return all sampled
            return sampled

        keep_mask = keep_mask.nonzero().squeeze(1)
        sampled[keep_mask] = inp[keep_mask]

    return sampled
