from __future__ import annotations

import torch



def planner_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    pred_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
    pred_stop_logit: torch.Tensor,
    target_stop: torch.Tensor,
    waypoint_l1_weight: float = 1.0,
    anchor_weight: float = 0.5,
    stop_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    waypoint_l1 = torch.nn.functional.l1_loss(pred_waypoints, target_waypoints)
    anchor_l1 = torch.nn.functional.l1_loss(pred_anchor, target_anchor)
    stop_bce = torch.nn.functional.binary_cross_entropy_with_logits(pred_stop_logit, target_stop)
    loss = waypoint_l1_weight * waypoint_l1 + anchor_weight * anchor_l1 + stop_weight * stop_bce
    return loss, {
        "waypoint_l1": float(waypoint_l1.detach().cpu()),
        "anchor_l1": float(anchor_l1.detach().cpu()),
        "stop_bce": float(stop_bce.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
