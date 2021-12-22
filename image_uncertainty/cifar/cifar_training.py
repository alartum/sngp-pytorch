import argparse
import datetime
import os
import re
import sys
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from ..models import get_model
from . import settings
from .cifar_datasets import get_test_dataloader, get_training_dataloader

# from face_uncertainty.models import get_model
# from face_uncertainty.cifar.cifar_datasets import get_test_dataloader, get_training_dataloader
# from face_uncertainty.cifar import settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="resnet50", help="net type")
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="use gpu or not"
    )
    parser.add_argument(
        "-b", type=int, default=128, help="batch size for dataloader"
    )
    parser.add_argument(
        "--warm", type=int, default=1, help="warm up training phase"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="initial learning rate"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="resume training"
    )
    parser.add_argument(
        "--weights", type=str, default="", help="model checkpoint weights"
    )
    parser.add_argument(
        "--ood-name", type=str, default="lsun", help="name of the data split"
    )
    parser.add_argument(
        "--data-seed", type=int, default=42, help="fixing the train/val sets"
    )
    args = parser.parse_args()

    return args


def train_cifar():
    args = get_args()
    model_path = train_model(args)
    print("*****")
    print(model_path)
    print("*****")


def train_model(args):
    model = get_model(args.net, args.gpu)

    # data preprocessing:
    training_loader, val_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=8,
        batch_size=args.b,
        shuffle=True,
        ood_name=args.ood_name,
        seed=args.data_seed,
        val_size=0.8,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES, gamma=0.2
    )  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_dir = os.path.join(
        settings.CHECKPOINT_PATH, args.net, f"{args.ood_name}_{args.data_seed}"
    )

    if args.resume:
        recent_folder = most_recent_folder(
            checkpoint_dir, fmt=settings.DATE_FORMAT
        )
        if not recent_folder:
            raise Exception("no recent folder were found")

        checkpoint_path = os.path.join(checkpoint_dir, recent_folder)

    else:
        checkpoint_path = os.path.join(checkpoint_dir, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # #since tensorboard can't overwrite old values
    # #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(1, 3, 32, 32)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(model, input_tensor)
    writer = None

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")

    best_acc = -1

    if args.resume:
        best_weights = best_acc_weights(
            os.path.join(checkpoint_dir, recent_folder)
        )
        if best_weights:
            weights_path = os.path.join(
                checkpoint_dir, recent_folder, best_weights
            )
            print("found best acc weights file:{}".format(weights_path))
            print("load best training file to test acc...")
            model.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print("best acc is {:0.2f}".format(best_acc))

        recent_weights_file = most_recent_weights(
            os.path.join(checkpoint_dir, recent_folder)
        )
        if not recent_weights_file:
            raise Exception("no recent weights file were found")
        weights_path = os.path.join(
            checkpoint_dir, recent_folder, recent_weights_file
        )
        print(
            "loading weights file {} to resume training.....".format(
                weights_path
            )
        )
        model.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(checkpoint_dir, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(
            epoch,
            model,
            args,
            optimizer,
            loss_function,
            training_loader,
            writer,
            warmup_scheduler,
        )
        acc = 2.0
        # acc = eval_training(
        #     epoch, True, model, val_loader, args.gpu, loss_function, writer
        # )

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type="best"
            )
            print("saving weights file to {}".format(weights_path))
            torch.save(model.state_dict(), weights_path)
            best_acc = acc
            continue

        if epoch % settings.SAVE_EPOCH == 0:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type="regular"
            )
            print("saving weights file to {}".format(weights_path))
            torch.save(model.state_dict(), weights_path)

    # writer.close()

    checkpoint_dir = Path(checkpoint_path).parent
    best_weights = best_acc_weights(checkpoint_dir)

    return checkpoint_dir / best_weights


def train(
    epoch,
    model,
    args,
    optimizer,
    loss_function,
    training_loader,
    writer,
    warmup_scheduler,
):
    start = time()
    model.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        # last_layer = list(model.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print(
            "Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                loss.item(),
                optimizer.param_groups[0]["lr"],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(training_loader.dataset),
            )
        )

        # #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    # for name, param in model.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time()

    print(
        "epoch {} training time consumed: {:.2f}s".format(
            epoch, finish - start
        )
    )


torch.no_grad()


def eval_training(epoch, tb, model, test_loader, gpu, loss_function, writer):
    start = time()
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    with torch.no_grad():
        for (images, labels) in test_loader:
            if gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    finish = time()
    if gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            test_loss / len(test_loader.dataset),
            correct.float() / len(test_loader.dataset),
            finish - start,
        )
    )
    print()

    # add informations to tensorboard
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


def restore_model(path, args):
    pass


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def most_recent_folder(net_weights, fmt):
    """
    return most recent created folder under net_weights
    if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [
        f for f in folders if len(os.listdir(os.path.join(net_weights, f)))
    ]
    if len(folders) == 0:
        return ""

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
    return most recent created weights file
    if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"

    # sort files by epoch
    weight_files = sorted(
        weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1])
    )

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception("no recent weights were found")
    resume_epoch = int(weight_file.split("-")[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
    return the best acc .pth file in given folder, if no
    best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"
    best_files = [
        w for w in files if re.search(regex_str, w).groups()[2] == "best"
    ]
    if len(best_files) == 0:
        return ""

    best_files = sorted(
        best_files, key=lambda w: int(re.search(regex_str, w).groups()[1])
    )
    return best_files[-1]


if __name__ == "__main__":
    train_cifar()
