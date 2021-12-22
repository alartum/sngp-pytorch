import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    # gradient_penalty = (grad_norm ** 2).mean()

    return gradient_penalty


def benchmark(
    dl_test, model_, epoch=0, loss=0, num_classes=100, l_gradient_penalty=0
):
    with torch.no_grad():
        x, y = next(iter(dl_test))
        x = x.cuda()
        y = y.cuda()
        # x.requires_grad_(True)
        y_pred = model_(x, debug=True)
        y = F.one_hot(y, num_classes).float()
        accuracy = (
            torch.sum(torch.argmax(y, dim=-1) == torch.argmax(y_pred, dim=-1))
            / len(y)
        ).item()
        bce = F.binary_cross_entropy(y_pred, y).item()
        if l_gradient_penalty > 0:
            gp = l_gradient_penalty * calc_gradient_penalty(model_.z, y_pred)
        else:
            gp = 0

    print(f"{epoch}: {accuracy:.3f}, {bce:.3f}, {gp:.3f}, {loss:.3f}")
    return accuracy


# def calc_gradient_penalty(x, y_pred):
#     gradients = torch.autograd.grad(
#         outputs=y_pred,
#         inputs=x,
#         grad_outputs=torch.ones_like(y_pred),
#         create_graph=True,
#     )[0]
#
#     gradients = gradients.flatten(start_dim=1)
#
#     # L2 norm
#     grad_norm = gradients.norm(2, dim=1)
#
#     # Two sided penalty
#     gradient_penalty = ((grad_norm - 1) ** 2).mean()
#
#     # One sided penalty - down
#     #     gradient_penalty = F.relu(grad_norm - 1).mean()
#
#     return gradient_penalty
#
#
# def benchmark(dl_test, model, epoch=0, loss=0, l_gradient_penalty=0):
#     x, y = next(iter(dl_test))
#     x = x.cuda()
#     y = y.cuda()
#     x.requires_grad_(True)
#     y_pred = model(x)
#     num_classes = y_pred.shape[-1]
#     y = F.one_hot(y, num_classes).float()
#     accuracy = (torch.sum(torch.argmax(y, dim=-1) == torch.argmax(y_pred, dim=-1)) / len(y)).item()
#     bce = F.binary_cross_entropy(y_pred, y).item()
#     if l_gradient_penalty:
#         gp = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
#     else:
#         gp = 0
#
#     print(f"{epoch}: {accuracy:.3f}, {bce:.3f}, {gp:.3f}, {loss:.3f}")
#     return accuracy


class DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.feature_extractor(x)
        self.z = z
        y_pred = self.rbf(z)

        return y_pred


class Head(nn.Module):
    def __init__(self, features, num_embeddings, sigma, embeddings_size):
        super().__init__()

        self.gamma = 0.99
        self.sigma = sigma

        self.W = nn.Parameter(
            torch.normal(
                torch.zeros(embedding_size, num_embeddings, features), 1
            )
        )

        self.register_buffer("N", torch.ones(num_embeddings) * 20)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_embeddings), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        # i is batch, m is embedding_size, n is num_embeddings (classes)
        x = torch.einsum("ij,mnj->imn", x, self.W)
        return x

    def bilinear(self, z):
        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        y_pred = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()
        return y_pred

    def forward(self, x):
        z = self.embed(x)
        y_pred = self.bilinear(z)

        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = torch.max(
            self.gamma * self.N + (1 - self.gamma) * y.sum(0),
            torch.ones_like(self.N),
        )

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


class MultiLinearCentroids(nn.Module):
    def __init__(
        self,
        num_classes=100,
        gamma=0.999,
        embedding_size=2048,
        features=128,
        feature_extractor=None,
        batch_size=128,
        sigma=1,
    ):
        super().__init__()
        # any number to initialize batching, ~batch_size / number classes
        self.N = (torch.ones(num_classes) * batch_size / num_classes).cuda()
        self.gamma = gamma
        # self.linear = spectral_norm(nn.Linear(embedding_size, centroids))
        self.feature_extractor = feature_extractor
        self.linears = nn.ModuleList(
            [
                spectral_norm(nn.Linear(embedding_size, features))
                for _ in range(num_classes)
            ]
        )

        self.means = torch.zeros((num_classes, features)).cuda()
        self.c = torch.zeros((num_classes, features)).cuda()
        self.sigma = sigma

    def _W(self, x):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        mappings = torch.stack(tuple([l(x) for l in self.linears]))
        z = torch.transpose(mappings, 0, 1)
        return z

    def forward(self, x, debug=False):
        z = self._W(x)
        probs = self.rbf(z, debug)
        return probs

    def rbf(self, x, debug=False):
        dist = torch.norm(self.c[None, :, :] - x, dim=-1)
        K = -(dist ** 2) / (2 * self.sigma ** 2)
        probs = torch.exp(K)
        if debug:
            print(dist[:3, :3])
            print("Dist", -torch.topk(-dist[:3], 5)[0])
            print("Dist", torch.topk(dist[:3], 5)[0])
            print("Probs", torch.topk(probs, 6, dim=-1)[0][:5])
        return probs

    def update_embeddings(self, x, y):
        with torch.no_grad():
            z = self._W(x)
            self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
            embedding_sum = torch.einsum("bcf,bc->cf", z, y)
            self.means = (
                self.gamma * self.means + (1 - self.gamma) * embedding_sum
            )
            self.c = self.means / self.N.unsqueeze(-1)


class LinearCentroids(nn.Module):
    def __init__(
        self,
        num_classes=100,
        gamma=0.999,
        embedding_size=2048,
        features=128,
        feature_extractor=None,
        batch_size=128,
        sigma=1,
    ):
        super().__init__()
        # any number to initialize batching, ~batch_size / number classes
        self.N = (torch.ones(num_classes) * batch_size / num_classes).cuda()
        self.gamma = gamma
        self.linear = spectral_norm(nn.Linear(embedding_size, features))
        self.feature_extractor = feature_extractor

        self.means = torch.zeros((num_classes, features)).cuda()
        self.c = torch.zeros((num_classes, features)).cuda()
        self.sigma = sigma

    def _W(self, x):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        z = self.linear(x)
        return z

    def forward(self, x, debug=False):
        z = self._W(x)
        probs = self.rbf(z, debug)
        return probs

    def rbf(self, x, debug=False):
        distances = torch.norm(self.c[None, :, :] - x[:, None, :], dim=-1)
        K = -(distances ** 2) / (2 * self.sigma ** 2)
        probs = torch.exp(K)
        if debug:
            print(distances[:3, :3])
            print("Dist", -torch.topk(-distances[:3], 5)[0])
            print("Dist", torch.topk(distances[:3], 5)[0])
            print("Probs", torch.topk(probs, 6, dim=-1)[0][:5])
        return probs

    def update_embeddings(self, x, y):
        with torch.no_grad():
            z = self._W(x)
            self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
            embedding_sum = torch.einsum("bf,bc->cf", z, y)
            self.means = (
                self.gamma * self.means + (1 - self.gamma) * embedding_sum
            )
            self.c = self.means / self.N.unsqueeze(-1)
