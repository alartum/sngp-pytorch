import sys

import torch


def get_model(architecture, gpu, **kwargs):
    """return given network"""
    if architecture == "resnet18":
        from .resnet import resnet18

        model = resnet18()
    elif architecture == "resnet34":
        from .resnet import resnet34

        model = resnet34()
    elif architecture == "resnet50":
        if "mc_dropout" in kwargs and kwargs["mc_dropout"]:
            from .resnet import resnet50_dropout

            model = resnet50_dropout(kwargs["dropout_rate"])
        else:
            from .resnet import resnet50

            model = resnet50()
    elif architecture == "resnet50_spectral":
        from spectral_normalized_models.resnet import ResNet50

        model = ResNet50(use_sn=True, num_classes=100)
    elif architecture == "resnet101":
        from .resnet import resnet101

        model = resnet101()
    elif architecture == "resnet152":
        from .resnet import resnet152

        model = resnet152()
    elif architecture == "wide_resnet50":
        model = torch.hub.load(
            "pytorch/vision:v0.6.0", "wide_resnet50_2", pretrained=True
        )

    elif architecture.startswith("wrn"):
        from .wrn import Wide_ResNet

        if "dropout_rate" not in kwargs:
            kwargs["dropout_rate"] = 0.1
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = 100
        if architecture == "wrn16_4":
            model = Wide_ResNet(
                16,
                4,
                dropout_rate=kwargs["dropout_rate"],
                num_classes=kwargs["num_classes"],
            )
        elif architecture == "wrn28_4":
            model = Wide_ResNet(
                28,
                4,
                dropout_rate=kwargs["dropout_rate"],
                num_classes=kwargs["num_classes"],
            )
        elif architecture == "wrn28_10":
            model = Wide_ResNet(
                28,
                10,
                dropout_rate=kwargs["dropout_rate"],
                num_classes=kwargs["num_classes"],
            )
    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    if gpu:
        model = model.cuda()

    return model
