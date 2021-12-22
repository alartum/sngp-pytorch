import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def t_scaling(logits, temperature):
    return torch.div(logits, temperature)


def calc_bins(probs, labels):
    num_bins = 20
    confs = np.max(probs, axis=-1)
    correct = np.argmax(probs, axis=-1) == labels
    print(correct)
    bins = np.linspace(1 / num_bins, 1, num_bins)
    binned = np.digitize(confs, bins)

    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        mask = binned == bin
        bin_sizes[bin] = np.sum(mask)
        print(bin, bin_sizes[bin])

        if bin_sizes[bin] > 0:
            bin_accs[bin] = correct[mask].sum() / bin_sizes[bin]
            bin_confs[bin] = confs[mask].sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def expected_ce(probs, labels):
    ECE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(probs, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    return ECE


def maximum_ce(probs, labels):
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(probs, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        MCE = max(MCE, abs_conf_dif)

    return MCE


def get_metrics(probs, labels):
    ece, mce = expected_ce(probs, labels), maximum_ce(probs, labels)
    return ece, mce


def draw_reliability_graph(preds, labels, name="", stage="uncalibrated"):
    ECE, MCE = get_metrics(preds, labels)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")

    # Error bars
    plt.bar(
        bins,
        bins,
        width=0.05,
        alpha=0.3,
        edgecolor="black",
        color="r",
        hatch="\\",
    )

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.05, alpha=1, edgecolor="black", color="b")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect("equal", adjustable="box")

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(
        color="green", label="ECE = {:.2f}%".format(ECE * 100)
    )
    MCE_patch = mpatches.Patch(
        color="red", label="MCE = {:.2f}%".format(MCE * 100)
    )
    plt.legend(handles=[ECE_patch, MCE_patch])
    plt.title(
        f"Model calibration (resnet18, cifar100 without {name}), {stage}"
    )

    plt.savefig(
        f"figures/calibrated_network_{name}_{stage}.png",
        bbox_inches="tight",
        dpi=80,
    )
    plt.show()


def scale_temperature(model, val_loader):
    device = torch.device("cuda")

    temperature = nn.Parameter(torch.ones(1).cuda())
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = torch.optim.LBFGS(
        [temperature], lr=0.001, max_iter=10000, line_search_fn="strong_wolfe"
    )

    logits_list = []
    labels_list = []
    temps = []
    losses = []

    for i, data in enumerate(tqdm(val_loader, 0)):
        images, labels = data[0].to(device), data[1].to(device)

        model.eval()
        with torch.no_grad():
            logits_list.append(model(images))
            labels_list.append(labels)

    # Create tensors
    logits_list = torch.cat(logits_list).to(device)
    labels_list = torch.cat(labels_list).to(device)

    def _eval():
        loss = criterion(t_scaling(logits_list, temperature), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss

    optimizer.step(_eval)

    print("Final T_scaling factor: {:.2f}".format(temperature.item()))

    plt.subplot(121)
    plt.plot(list(range(len(temps))), temps)

    plt.subplot(122)
    plt.plot(list(range(len(losses))), losses)
    plt.show()

    return temperature.item()


class WrappedModel:
    def __init__(self, model, T):
        self.T = T
        self.model = model

    def __call__(self, x):
        return t_scaling(self.model(x), self.T)
