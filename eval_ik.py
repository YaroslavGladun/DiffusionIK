import torch
import argparse
from math import pi

from fk import FK
from common import JointValuesScalerInverse


def geodesic_distance(R_pred, R_target):
    """Geodesic distance on SO(3) in radians."""
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_target)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    return torch.acos(cos_angle)


def evaluate_ik(
    predict_fn,
    device,
    num_targets=1000,
    samples_per_target=50,
    position_threshold_m=0.001,
    orientation_threshold_rad=1.0 * pi / 180,
):
    """
    Evaluate an IK model on two metrics:
      1. Success rate — fraction of targets where at least one sample
         is within position and orientation thresholds.
      2. Diversity — mean pairwise joint-space distance (radians)
         among successful samples, averaged over targets.

    Args:
        predict_fn: callable(R_target [1,3,3], t_target [1,3,1], n) -> joints [n, 7]
            Called once per target. Must return `n` joint configurations (radians).
        device: torch device
        num_targets: number of random reachable EE targets to evaluate
        samples_per_target: how many IK solutions to request per target
        position_threshold_m: success threshold for position error (meters)
        orientation_threshold_rad: success threshold for orientation error (radians)

    Returns:
        dict with 'success_rate' and 'diversity'
    """
    fk = FK(device)
    scaler_inv = JointValuesScalerInverse(device)

    successes = 0
    diversities = []

    for i in range(num_targets):
        # random reachable target
        rand_joints = scaler_inv(torch.rand(1, 7, device=device))
        with torch.no_grad():
            R_target, t_target = fk(rand_joints)

        # model predictions
        with torch.no_grad():
            pred_joints = predict_fn(R_target, t_target, samples_per_target)
            R_pred, t_pred = fk(pred_joints)

        # errors
        pos_err = torch.norm(
            t_pred.squeeze(-1) - t_target.squeeze(-1), dim=-1
        )
        ori_err = geodesic_distance(R_pred, R_target.expand_as(R_pred))

        success_mask = (pos_err < position_threshold_m) & (
            ori_err < orientation_threshold_rad
        )
        successes += success_mask.any().item()

        # diversity among successful solutions
        successful_joints = pred_joints[success_mask]
        if successful_joints.shape[0] >= 2:
            diff = successful_joints.unsqueeze(0) - successful_joints.unsqueeze(1)
            pairwise = torch.norm(diff, dim=-1)
            triu_mask = torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool), diagonal=1
            )
            diversities.append(pairwise[triu_mask].mean().item())

    success_rate = successes / num_targets
    diversity = sum(diversities) / len(diversities) if diversities else 0.0

    return {"success_rate": success_rate, "diversity": diversity}


# ---- example: random baseline (replace with your model) ----

def random_predict_fn(device):
    """Returns a predict_fn that outputs random joints within limits."""
    scaler_inv = JointValuesScalerInverse(device)

    def predict(R_target, t_target, n):
        return scaler_inv(torch.rand(n, 7, device=device))

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IK model")
    parser.add_argument("--num-targets", type=int, default=1000)
    parser.add_argument("--samples-per-target", type=int, default=50)
    parser.add_argument("--pos-thresh", type=float, default=0.001,
                        help="Position threshold in meters")
    parser.add_argument("--ori-thresh", type=float, default=1.0,
                        help="Orientation threshold in degrees")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # swap this with your trained model's predict_fn
    predict_fn = random_predict_fn(device)

    results = evaluate_ik(
        predict_fn,
        device,
        num_targets=args.num_targets,
        samples_per_target=args.samples_per_target,
        position_threshold_m=args.pos_thresh,
        orientation_threshold_rad=args.ori_thresh * pi / 180,
    )

    print(f"Success rate: {results['success_rate']:.4f} "
          f"({results['success_rate'] * 100:.2f}%)")
    print(f"Diversity:    {results['diversity']:.4f} rad")
