import os
from pathlib import Path

import gpytorch
import numpy as np
import torch
import tqdm
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from ..models.svdkl import (
    DenseNetFeatureExtractor,
    DKLModel,
    GaussianProcessLayer,
)
from . import settings
from .cifar_datasets import get_test_dataloader, get_training_dataloader


def train_svkdl(args):
    training_loader, val_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        ood_name=args.ood_name,
    )

    checkpoint_dir = os.path.join(
        settings.CHECKPOINT_PATH, args.net, args.ood_name
    )
    checkpoint_path = Path(os.path.join(checkpoint_dir, settings.TIME_NOW))
    print(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    num_classes = 100

    feature_extractor = DenseNetFeatureExtractor(
        block_config=(6, 6, 6), num_classes=num_classes
    )
    num_features = feature_extractor.classifier.in_features

    model = DKLModel(feature_extractor, num_dim=num_features)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
        num_features=model.num_dim, num_classes=num_classes
    )

    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    n_epochs = 100
    lr = 0.1
    optimizer = SGD(
        [
            {
                "params": model.feature_extractor.parameters(),
                "weight_decay": 1e-4,
            },
            {"params": model.gp_layer.hyperparameters(), "lr": lr * 0.01},
            {"params": model.gp_layer.variational_parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=0,
    )
    scheduler = MultiStepLR(
        optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1
    )
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model.gp_layer, num_data=len(training_loader.dataset)
    )

    for epoch in range(1, n_epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            train_epoch(
                model, likelihood, training_loader, optimizer, mll, epoch
            )
            test_epoch(model, likelihood, val_loader)
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save(
            {"model": state_dict, "likelihood": likelihood_state_dict},
            checkpoint_path / "dkl_checkpoint.dat",
        )

    return checkpoint_path / "dkl_checkpoint.dat"


def train_epoch(model, likelihood, train_loader, optimizer, criterion, epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())


def test_epoch(model, likelihood, test_loader, return_runs=False):
    model.eval()
    likelihood.eval()

    correct = []
    runs = None

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(
                model(data)
            )  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(
                -1
            )  # Taking the mean over all of the sample we've drawn
            correct = np.concatenate(
                (correct, pred.eq(target.view_as(pred)).cpu().numpy())
            )

            if return_runs:
                batch_runs = output.probs.detach().cpu().numpy()
                if runs is None:
                    runs = output.probs.detach().cpu().numpy()
                else:
                    runs = np.concatenate((runs, batch_runs), axis=1)

    print(
        "Test set: Accuracy: {}/{} ({}%)".format(
            correct.sum(),
            len(test_loader.dataset),
            100.0 * correct.sum() / float(len(test_loader.dataset)),
        )
    )
    return runs, correct
