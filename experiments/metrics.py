from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"


def uncertainty_plot(
    ues,
    ood_ues,
    accuracy=None,
    title="Uncertainty CIFAR100",
    directory=FIGURES_DIR,
    file_name="ood_boxplot",
    show=False,
):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Uncertainty": np.hstack((ues, ood_ues)),
            "Type": np.hstack((["InD"] * len(ues), ["OOD"] * len(ood_ues))),
        }
    )

    plt.rc("font", size=14)
    plt.figure(figsize=(9, 7))
    sns.boxplot(x="Type", y="Uncertainty", data=df)
    plt.title(title)

    patches = []
    ood_score = ood_roc_auc(ues, ood_ues)
    if accuracy is not None:
        patches.append(Patch(color="none", label=f"InD accuracy {accuracy}"))
        print("Accuracy", accuracy)
    print("OOD ROC AUC", ood_score)
    patches.append(Patch(color="none", label=f"OOD roc-auc {ood_score}"))
    plt.legend(
        handles=patches, handlelength=0, handletextpad=0, loc="upper left"
    )

    if show:
        plt.show()
    else:
        plt.savefig(directory / f"{file_name}.png", dpi=150)

    return ood_score


def boxplots(
    ues,
    ood_ues,
    ood_name,
    extras="",
    show=False,
    directory=FIGURES_DIR,
    title_extras="",
):
    df = pd.DataFrame(
        {
            "Uncertainty": np.hstack((ues, ood_ues)),
            "Type": np.hstack((["InD"] * len(ues), ["OOD"] * len(ood_ues))),
        }
    )

    plt.rc("font", size=14)
    plt.figure(figsize=(12, 10))
    sns.boxplot(x="Type", y="Uncertainty", data=df)

    plt.title(f"Uncertainty on CIFAR100 ({ood_name} OOD){title_extras}")

    if show:
        plt.show()
    else:
        plt.savefig(
            directory / f"ood_boxplot_{extras}_{ood_name}.png", dpi=100
        )


def scatterplots(ues, ood_ues, ood_name, show=False, directory=FIGURES_DIR):
    plt.rc("font", size=14)
    plt.figure(figsize=(12, 10))
    alpha = 0.1
    size = 50
    gibberish = np.random.random(len(ues)) * 0.05
    plt.scatter(gibberish + 0.1, ues, alpha=alpha, s=size)
    plt.scatter(gibberish + 0.2, ood_ues, alpha=alpha, s=size)

    if show:
        plt.show()
    else:
        plt.savefig(directory / f"ood_scatterplot_{ood_name}.png", dpi=100)


def ood_roc_auc(ues, ood_ues):
    labels = np.concatenate((np.zeros_like(ues), np.ones_like(ood_ues)))
    scores = np.concatenate((ues, ood_ues))
    return roc_auc_score(labels, scores).round(3)


def count_alphas(ues, ood_ues, show=False):
    correct_ue = np.sum(np.array(ues) < 0.1)
    correct_ood = np.sum(np.array(ood_ues) > 0.9)

    print(
        f"Correct InD: {correct_ue} out of {len(ues)} ({correct_ue/len(ues)})"
    )
    print(
        f"Correct OOD: {correct_ood} out of {len(ood_ues)} "
        f"({correct_ood/len(ood_ues)})"
    )
    alpha = (correct_ue + correct_ood) / (len(ues) + len(ood_ues))
    print(f"Alpha is {alpha}")

    splits = np.linspace(0, 1, 100)
    alphas = []
    for split in splits:
        correct_ue = np.sum(np.array(ues) < split)
        correct_ood = np.sum(np.array(ood_ues) > split)
        alphas.append((correct_ue + correct_ood) / (len(ues) + len(ood_ues)))
    print("Best alpha", np.max(alphas))
    if show:
        plt.plot(splits, alphas)
        plt.title(f"Best alpha is {np.max(alphas)}")
        plt.show()

    return alpha, np.max(alphas)
