from math import ceil

import numpy as np
import torch
from tqdm import tqdm


def maxprob_ue(loader, model, gpu, acquisition="max_prob"):
    ues = []
    correct = []

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(loader):
            if (n_iter + 1) % ceil(len(loader)) == 0:
                print(
                    "iteration: {}\ttotal {} iterations".format(
                        n_iter + 1, len(loader)
                    )
                )
            if gpu:
                image = image.cuda()
                label = label.cuda()
            output = model(image)
            vals, pred = torch.softmax(output, dim=-1).topk(1)
            correct.extend((pred[:, 0] == label).cpu().tolist())

            if acquisition == "max_prob":
                ue = 1 - vals
                ues.extend(ue[:, 0].cpu().tolist())
            elif acquisition == "entropy":
                ue = entropy(torch.softmax(output, dim=-1).cpu().numpy())
                ues.extend(list(ue))
            else:
                raise ValueError(
                    f"Value of metric ({acquisition}) is incorrect"
                )

    return ues, correct


def mcd_runs(loader, model, gpu, repeats=None, ensemble=False, last_iter=None):
    runs = None
    correct = []

    with torch.no_grad():
        for n_iter, (images, labels) in enumerate(tqdm(loader)):
            if last_iter is not None and last_iter == n_iter:
                break
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
            if ensemble:
                preds = torch.stack([m(images) for m in model])
            else:
                preds = torch.stack([model(images) for _ in range(repeats)])

            softmaxed = torch.softmax(preds, dim=-1)
            averaged = torch.mean(softmaxed, dim=0)
            _, pred = torch.softmax(averaged, dim=-1).topk(1)
            correct.extend((pred[:, 0] == labels).cpu().tolist())

            batch_runs = softmaxed.detach().cpu().numpy()
            if runs is None:
                runs = batch_runs
            else:
                runs = np.concatenate((runs, batch_runs), axis=1)

    return runs, correct


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-6, 1)), axis=-1)


def bald(probabilities):
    predictive_entropy = entropy(np.mean(probabilities, axis=0))
    expected_entropy = np.mean(entropy(probabilities), axis=0)
    res = predictive_entropy - expected_entropy
    return res


def mcd_ue(runs, acquisition="max_prob"):
    if acquisition == "max_prob":
        ue = 1 - np.max(np.mean(runs, axis=0), axis=-1)
    elif acquisition == "entropy":
        ue = entropy(np.mean(runs, axis=0))
    elif acquisition == "bald":
        ue = bald(runs)
    elif acquisition == "std":
        stds = np.std(runs, axis=0)
        idxs = np.argmax(runs.mean(axis=0), axis=-1)
        ue = stds[np.arange(len(idxs)), idxs]
    else:
        raise ValueError("Unknown acquisition")

    return ue
