import torch
import torch.nn.functional as F


def policy_loss_fn(
    policy_logits,
    policy_targets
):
    return F.cross_entropy(
        policy_logits,
        policy_targets
    )


def value_loss_fn(
    value_pred,
    value_target
):
    return F.smooth_l1_loss(
        value_pred.squeeze(-1),
        value_target.squeeze(-1)
    )


def total_loss(
    policy_logits,
    policy_targets,
    value_pred,
    value_target,
    value_weight=1.0
):

    p_loss = policy_loss_fn(
        policy_logits,
        policy_targets
    )

    v_loss = value_loss_fn(
        value_pred,
        value_target
    )

    return (
        p_loss +
        value_weight * v_loss,
        p_loss,
        v_loss
    )