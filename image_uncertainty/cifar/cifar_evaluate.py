import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..models import get_model
from ..uncertainty.methods import maxprob_ue
from ..uncertainty.metrics import boxplots, count_alphas, uncertainty_plot
from . import settings
from .cifar_datasets import get_test_dataloader


def default_weights(net, ood_name, data_seed, num=0):
    if ood_name in ["svhn", "lsun", "isun", "cifar10", "smooth"]:
        # it's the checkpoint for full cifar100 dataset models (historically ¯\_(ツ)_/¯)
        ood_name = "lsun"
    return (
        f"experiments/checkpoint/{net}/{ood_name}_{data_seed}/model_{num}.pth"
    )


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="resnet50", help="net type")
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="use gpu or not"
    )
    parser.add_argument(
        "-b", type=int, default=128, help="batch size for dataloader"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="model checkpoint weights"
    )
    parser.add_argument(
        "--ood-name",
        type=str,
        default="vehicles",
        help="model checkpoint weights",
    )
    parser.add_argument(
        "--data-seed", type=int, default=42, help="fixing the train/val sets"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="use cached representations",
    )
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument(
        "--mode", type=str, default="logits", help="logits or features"
    )
    args = parser.parse_args()

    if args.weights is None:
        args.weights = default_weights(args.net, args.ood_name, args.data_seed)

    return args


def cifar_test(batch_size, ood, ood_name, subsample=None):
    loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        ood=ood,
        ood_name=ood_name,
        subsample=subsample,
    )
    return loader


def load_model(architecture, weights, gpu, **kwargs):
    model = get_model(architecture, gpu, **kwargs)
    model.load_state_dict(torch.load(weights))
    model.eval()
    return model


def inference(model, val_loader, gpu):
    preds = []
    labels = []
    probs = []
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(val_loader):
            print(
                "iteration: {}\ttotal {} iterations".format(
                    n_iter + 1, len(val_loader)
                )
            )

            if gpu:
                image = image.cuda()
                label = label.cuda()

            output = model(image)
            output = torch.softmax(output, dim=-1)
            conf, prediction = output.topk(1)

            probs.extend(output.tolist())
            preds.extend(prediction.cpu().tolist())
            labels.extend(label.cpu().tolist())
            # break
    probs = np.array(probs)
    labels = np.array(labels)
    return preds, probs, labels


def described_plot(
    ues,
    ood_ues,
    ood_name,
    net,
    accuracy=None,
    title_extras="",
    dataset_name="CIFAR100",
    show=False,
):
    label = f"{net}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    file_name_extras = title_extras.replace(" ", "-")
    file_name = f"ood_boxplot__{ood_name}_{label}_{file_name_extras}"
    title = f"Uncertainty on {dataset_name} ({ood_name} OOD) {title_extras}"

    return uncertainty_plot(
        ues,
        ood_ues,
        title=title,
        file_name=file_name,
        accuracy=accuracy,
        show=show,
    )


def dump_ues(ues, ood_ues, method, dataset_name, ood_name):
    base_dir = Path("checkpoint") / dataset_name
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    name = f"{method}_{ood_name}".replace(" ", "_")

    with open(base_dir / f"{name}_ues.npy", "wb") as f:
        np.save(f, np.array(ues))

    with open(base_dir / f"{name}_ood_ues.npy", "wb") as f:
        np.save(f, np.array(ood_ues))


from sklearn.metrics import roc_auc_score


def misclassification_detection(correct, ue):
    try:
        label = (~np.array(correct)).astype(int)
        print("Misclassification detection roc auc")
        print(roc_auc_score(label, np.array(ue)))
    except:
        import ipdb

        ipdb.set_trace()


def evaluate(model, args, title_extras="", dataset_name="CIFAR100"):
    test_loader = cifar_test(args.b, False, args.ood_name)
    ood_loader = cifar_test(args.b, True, args.ood_name)

    _, targets = next(iter(test_loader))

    print("max_prob")
    ues, correct = maxprob_ue(test_loader, model, args.gpu, "max_prob")
    ood_ues, _ = maxprob_ue(ood_loader, model, args.gpu, "max_prob")
    accuracy = sum(correct) / len(correct)

    misclassification_detection(correct, ues)

    described_plot(
        ues,
        ood_ues,
        args.ood_name,
        args.net,
        accuracy,
        title_extras,
        dataset_name,
    )
    print("entropy")
    ues, correct = maxprob_ue(test_loader, model, args.gpu, "entropy")
    ues_ood, _ = maxprob_ue(ood_loader, model, args.gpu, "entropy")
    accuracy = sum(correct) / len(correct)

    misclassification_detection(correct, ues)

    described_plot(
        ues,
        ues_ood,
        args.ood_name,
        args.net,
        accuracy,
        title_extras,
        dataset_name,
    )

    dump_ues(ues, ues_ood, f"discrete_44", "cifar", args.ood_name)
