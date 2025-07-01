import torch
import torch.nn as nn

class TanhDirectionalLoss(nn.Module):
    def __init__(self, mse_weight=1.0, directional_weight=0.5, diversity_weight=1.0, smooth_scale=2.0, epoch=0, max_epochs=20):
        super().__init__()
        self.mse_weight = mse_weight
        self.base_directional_weight = directional_weight
        self.diversity_weight = diversity_weight
        self.smooth_scale = smooth_scale
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def update_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, outputs, targets):
        # Scale outputs to amplify gradient signal
        output_scale = 10.0
        scaled_outputs = outputs * output_scale
        scaled_targets = targets * output_scale

        # Normalize targets
        target_std = torch.std(scaled_targets) + 1e-8
        normalized_targets = scaled_targets / target_std
        normalized_outputs = scaled_outputs / target_std

        mse_loss = self.mse_weight * self.mse(normalized_outputs, normalized_targets)

        directional_weight = self.base_directional_weight * (self.epoch + 1) / self.max_epochs
        directional_loss = directional_weight * torch.mean(torch.relu(-outputs * torch.sign(targets)))

        pred_var = torch.var(outputs, dim=-1).mean()
        diversity_loss = self.diversity_weight * -torch.log(pred_var + 1e-6)

        total_loss = mse_loss + directional_loss + diversity_loss
        return total_loss


