import torch
import torch.nn.functional as F


class SoftKMeans:
    def __init__(self, n_iters: int = 10, eps: float = 1e-6, temperature: float = 0.1, repeat: int = 3):
        self.n_iters = n_iters
        self.eps = eps
        self.temperature = temperature
        self.repeat = repeat

        self.centers, self.cost, self.weights = None, None, None

    def reset(self):
        self.centers, self.cost, self.weights = None, None, None
    
    @torch.no_grad()
    def fit(self, features):
        """
        Args:
            features (torch.Tensor): shape [N, D]
        """
        N, D = features.shape
        best_cost = float('inf')
        # repeat for several times to avoid local optima
        for _ in range(self.repeat):
            # initialize centers
            center1 = features[torch.randint(0, N, (1, ))].squeeze(0)
            dist = 1 - torch.mm(center1[None, :], features.T).squeeze(0)
            center2 = features[torch.argmax(dist, dim=-1)]
            centers = torch.stack((center1, center2), dim=0)
            cost = float('inf')
            # iterations
            for it in range(self.n_iters):
                dists = 1 - torch.mm(features, centers.T)
                weights = F.softmax(-dists / self.temperature, dim=-1)
                centers = torch.mm(weights.T, features)
                centers = F.normalize(centers, dim=-1)
                newcost = torch.sum(weights * dists, dim=(-1, -2)).item()
                if (abs(newcost - cost) < self.eps) or ((abs(newcost - cost) / cost) < self.eps):
                    break
                cost = newcost
            dists = 1 - torch.mm(features, centers.T)
            weights = F.softmax(-dists / self.temperature, dim=-1)
            # calculate cost
            cost = torch.sum(weights * dists, dim=(-1, -2))
            if cost < best_cost:
                self.cost = best_cost = cost
                self.weights = weights[:, 0]
                self.centers = centers

    @torch.no_grad()
    def predict(self, features):
        dists = 1 - torch.mm(features, self.centers.T)
        weights = F.softmax(-dists / self.temperature, dim=-1)
        return weights[:, 0]
