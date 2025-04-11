import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDistributionLoss(nn.Module):
    """
    Custom loss function that considers both location accuracy and spatial distribution
    """
    def __init__(self, distribution_weight=0.3):
        super(SpatialDistributionLoss, self).__init__()
        self.distribution_weight = distribution_weight
        
    def forward(self, pred, target):
        """
        Parameters:
        -----------
        pred : torch.Tensor
            Predicted locations, shape [batch_size, 2] for lat/lon
        target : torch.Tensor
            Target locations, shape [batch_size, 2] for lat/lon
            
        Returns:
        --------
        loss : torch.Tensor
            Combined loss value
        """
        # Base MSE loss for location accuracy
        mse_loss = F.mse_loss(pred, target)
        
        # Distribution preservation component:
        # Calculate variance of predictions and targets
        pred_var = torch.var(pred, dim=0)
        target_var = torch.var(target, dim=0)
        
        # Penalize differences in spatial distribution (variance)
        variance_loss = F.mse_loss(pred_var, target_var)
        
        # Calculate covariance matrices to preserve spatial patterns
        pred_centered = pred - torch.mean(pred, dim=0)
        target_centered = target - torch.mean(target, dim=0)
        
        pred_cov = torch.matmul(pred_centered.t(), pred_centered) / (pred.size(0) - 1)
        target_cov = torch.matmul(target_centered.t(), target_centered) / (target.size(0) - 1)
        
        # Covariance loss (preserves spatial correlations)
        cov_loss = F.mse_loss(pred_cov, target_cov)
        
        # Combine losses
        distribution_loss = variance_loss + cov_loss
        total_loss = (1 - self.distribution_weight) * mse_loss + self.distribution_weight * distribution_loss
        
        return total_loss

# Alternative implementation: Mixture Density Network (MDN) loss
# This allows modeling multimodal spatial distributions
class MDNLoss(nn.Module):
    def __init__(self, n_components=5):
        super(MDNLoss, self).__init__()
        self.n_components = n_components
        
    def forward(self, params, target):
        """
        Parameters:
        -----------
        params : torch.Tensor
            MDN parameters: means, sigmas, and mixture weights
            Shape: [batch_size, n_components * 5]  (2 means + 2 sigmas + 1 weight per component)
        target : torch.Tensor
            Target locations, shape [batch_size, 2] for lat/lon
        """
        n_components = self.n_components
        
        # Reshape params into components
        pi = params[:, :n_components]  # mixture weights
        mu_x = params[:, n_components:2*n_components]  # means for x
        mu_y = params[:, 2*n_components:3*n_components]  # means for y
        sigma_x = params[:, 3*n_components:4*n_components]  # stdevs for x
        sigma_y = params[:, 4*n_components:5*n_components]  # stdevs for y
        
        # Apply softmax to mixture weights
        pi = F.softmax(pi, dim=1)
        
        # Apply softplus to sigmas to ensure they're positive
        sigma_x = F.softplus(sigma_x) + 1e-3
        sigma_y = F.softplus(sigma_y) + 1e-3
        
        # Extract targets
        x_target = target[:, 0].unsqueeze(1)  # [batch_size, 1]
        y_target = target[:, 1].unsqueeze(1)  # [batch_size, 1]
        
        # Calculate Gaussian probabilities for each component
        x_normal = torch.distributions.Normal(mu_x, sigma_x)
        y_normal = torch.distributions.Normal(mu_y, sigma_y)
        
        x_log_prob = x_normal.log_prob(x_target.expand_as(mu_x))
        y_log_prob = y_normal.log_prob(y_target.expand_as(mu_y))
        
        # Combine x and y log probabilities
        xy_log_prob = x_log_prob + y_log_prob  # [batch_size, n_components]
        
        # Apply mixture weights
        component_log_prob = xy_log_prob + torch.log(pi + 1e-8)
        
        # Log sum exp trick for numerical stability
        max_comp_log_prob = torch.max(component_log_prob, dim=1, keepdim=True)[0]
        log_prob = max_comp_log_prob + torch.log(
            torch.sum(torch.exp(component_log_prob - max_comp_log_prob), dim=1, keepdim=True) + 1e-8
        )
        
        # Negative log likelihood
        loss = -torch.mean(log_prob)
        
        return loss