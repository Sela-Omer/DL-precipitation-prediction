import torch
from torchmetrics import Metric


class MAEWithConfidenceInterval(Metric):
    full_state_update = False

    def __init__(self, confidence_level=0.95):
        super().__init__()
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        self.confidence_level = confidence_level

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Compute absolute errors
        errors = torch.abs(preds - targets)
        self.errors.append(errors)

    def compute(self):
        # squeeze self.errors to remove empty dimensions
        all_errors = self.errors.squeeze(-1)

        # Compute the mean absolute error
        mae = torch.mean(all_errors)

        # Compute confidence interval
        n = len(all_errors)
        std_error = torch.std(all_errors) / torch.sqrt(torch.tensor(n, dtype=torch.float).to(device=all_errors.device))
        z = torch.tensor([1.96]).to(
            device=all_errors.device)  # Z-value for 95% confidence interval (normal distribution assumption)

        ci_lower = mae - z * std_error
        ci_upper = mae + z * std_error

        return {
            "mae_ci": mae,
            "mae_ci_lower_0.95": ci_lower,
            "mae_ci_upper_0.95": ci_upper,
        }
