import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def get_embedding(loaded_model, data, features, name="model.layer4"):
    with torch.no_grad():
        if features == "embeddings":
            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()

                return hook

            loaded_model.layer4.register_forward_hook(
                get_activation(name=name)
            )
            loaded_model(data)
            return activation[name]
        elif features == "logits":
            return loaded_model(data)
        else:
            raise ValueError("Incorrect type of features, used for GMM")


def receive_embeddings(
    model, dataloader, name="model.layer4", features="embeddings", device="cpu"
):
    embeddings = torch.tensor([])
    labels = torch.tensor([])
    print("Receiving embeddings..")
    for iter, batch in enumerate(tqdm(dataloader)):
        X = batch[0].to(device)
        y = batch[1]
        embedding = (
            get_embedding(
                loaded_model=model, data=X, name=name, features=features
            )
            .cpu()
            .detach()
        )
        embeddings = torch.cat([embeddings, embedding], dim=0)
        labels = torch.cat([labels, y])
    embedding_numpy = (
        embeddings.cpu().detach().numpy().reshape(embeddings.shape[0], -1)
    )
    labels = labels.cpu().detach().numpy()

    return embedding_numpy, labels


def fit_gmm_ddu_mf(
    model,
    dataloader,
    name="model.layer4",
    device="cpu",
    features="embeddings",
    embedding_numpy=None,
    labels=None,
):
    if embedding_numpy is None or labels is None:
        embedding_numpy, labels = receive_embeddings(
            model=model,
            dataloader=dataloader,
            name=name,
            device=device,
            features=features,
        )
    print("Fitting GMM..")
    means, covs = per_class_mean_covs(
        embeddings=embedding_numpy, labels=labels
    )
    gmm_density = GMM_density(locs=means, variances=covs)
    return gmm_density


def knn_ddu(
    model,
    dataloader,
    name="model.layer4",
    device="cpu",
    features="embeddings",
    embedding_numpy=None,
    labels=None,
):
    if embedding_numpy is None or labels is None:
        embedding_numpy, labels = receive_embeddings(
            model=model,
            dataloader=dataloader,
            name=name,
            device=device,
            features=features,
        )
    print("Fitting KNN..")
    knn = KNeighborsClassifier()
    knn.fit(X=embedding_numpy, y=labels)
    return knn


def per_class_mean_covs(embeddings, labels):
    means, covs = [], []
    for c in tqdm(np.unique(labels)):
        # print(c)
        mask = labels == c
        means.append(np.mean(embeddings[mask], axis=0))
        covs.append(np.var(embeddings[mask], axis=0))
    return np.array(means), np.array(covs)


def extract_data_from_dataloader(dataloader, device):
    full_objects = torch.tensor([], device=device)
    full_labels = torch.tensor([], device=device)
    for batch in dataloader:
        images, labels = batch
        full_objects = torch.cat([full_objects, images.to(device)], dim=0)
        full_labels = torch.cat([full_labels, labels.to(device)], dim=0)
    return full_objects, full_labels


def ddu_uncertainties_gmm(
    dataloader,
    model,
    gmm,
    name="model.layer4",
    device="cpu",
    features="embeddings",
):
    embeds = torch.tensor([])
    for batch in dataloader:
        data = batch[0].to(device)
        embed = (
            get_embedding(
                loaded_model=model, data=data, name=name, features=features
            )
            .cpu()
            .detach()
        )
        embeds = torch.cat([embeds, embed], dim=0)
    uncertainties = gmm.log_prob(embeds.view(embeds.shape[0], -1))
    return uncertainties, None


def ddu_uncertainties_knn(
    dataloader,
    model,
    knn,
    name="model.layer4",
    device="cpu",
    features="embeddings",
    train_labels=None,
    n_neighbors=5,
):
    embeds = torch.tensor([])
    for batch in dataloader:
        data = batch[0].to(device)
        embed = (
            get_embedding(
                loaded_model=model, data=data, name=name, features=features
            )
            .cpu()
            .detach()
        )
        embeds = torch.cat([embeds, embed], dim=0)
    embeds_numpy = embeds.cpu().detach().numpy()
    uncertainties = np.exp(
        -knn.kneighbors(X=embeds_numpy, n_neighbors=n_neighbors)[0]
    )
    denumenator = np.sum(embeds_numpy, axis=1)
    final_uncertainties = np.array([])
    n_classes = len(np.unique(train_labels))
    for c in np.unique(train_labels):
        mask = train_labels == c
        current_weights = (
            (np.sum(uncertainties[:, mask], axis=1) + 1)
            / (denumenator + 1.0 * n_classes)
        ).reshape(-1, 1)
        if final_uncertainties.shape[0] == 0:
            final_uncertainties = current_weights
        else:
            final_uncertainties = np.concatenate(
                [final_uncertainties, current_weights], axis=1
            )

    final_uncertainties = np.sum(
        final_uncertainties * np.log(1e-6 + final_uncertainties), axis=1
    )

    return final_uncertainties, None


class GMM_density:
    def __init__(self, locs, variances):
        self.locs = torch.tensor(locs)
        self.scales = torch.tensor(variances ** 0.5)
        self.distributions = []

        for i in tqdm(range(len(locs))):
            if torch.sum(self.scales[i]) == 0:
                self.scales[i] += 0.02  # to prevent from zeroing
            self.distributions.append(
                torch.distributions.Normal(
                    loc=self.locs[i], scale=self.scales[i]
                )
            )

    def log_prob(self, x):
        x = x.cpu().detach()
        log_p = torch.tensor([])
        for i in range(len(self.distributions)):
            log_paux = self.distributions[i].log_prob(x).sum(-1).view(
                -1, 1
            ) + torch.log(torch.tensor(1.0 / len(self.locs)))
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)
        return log_density

    def sample(self, shape):
        p = np.random.choice(a=len(self.distributions), size=shape[0])
        samples = torch.tensor([], device=self.device)
        for idx in p:
            z = self.distributions[idx].sample((1,))
            samples = torch.cat([samples, z])
        return samples
