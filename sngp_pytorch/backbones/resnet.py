# SOURCE https://github.com/stat-ml/image_uncertainty/blob/master/spectral_normalized_models/resnet.py # noqa E501

"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchinfo import summary


def SN_wrapper(layer, use_sn):
    if use_sn:
        return spectral_norm(layer)
    else:
        return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, use_sn, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SN_wrapper(
            nn.Conv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            use_sn=use_sn,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SN_wrapper(
            nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            ),
            use_sn=use_sn,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if not use_sn:
                self.shortcut = nn.Sequential(
                    SN_wrapper(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        use_sn=use_sn,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    SN_wrapper(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        use_sn=use_sn,
                    ),
                    nn.AvgPool2d(kernel_size=2, stride=stride)
                    if stride != 1
                    else nn.Sequential(),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, use_sn, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = SN_wrapper(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            use_sn=use_sn,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SN_wrapper(
            nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            use_sn=use_sn,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SN_wrapper(
            nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            ),
            use_sn=use_sn,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if not use_sn:
                self.shortcut = nn.Sequential(
                    SN_wrapper(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        use_sn=use_sn,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    SN_wrapper(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        use_sn=use_sn,
                    ),
                    nn.AvgPool2d(kernel_size=2, stride=stride)
                    if stride != 1
                    else nn.Sequential(),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        use_sn,
        num_ch=[64, 128, 256, 512],
        num_classes=10,
        input_planes=3,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_sn = use_sn
        if num_classes <= 100:  # cifar alike
            self.conv1 = SN_wrapper(
                nn.Conv2d(
                    input_planes,
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                use_sn=self.use_sn,
            )
            self.maxpool = nn.Sequential()  # void placeholder
        else:  # imagenet alike
            self.conv1 = SN_wrapper(
                nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                ),
                use_sn=self.use_sn,
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(
            block, num_ch[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, num_ch[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, num_ch[2], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, num_ch[3], num_blocks[3], stride=2
        )
        self.linear = SN_wrapper(
            nn.Linear(num_ch[3] * block.expansion, num_classes),
            use_sn=self.use_sn,
        )
        self.feature = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    use_sn=self.use_sn,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        out = self.linear(out)
        return out


class ResNetDropout(ResNet):
    pass

    def __init__(
        self,
        block,
        num_blocks,
        dropout_rate=0.0,
        use_sn=False,
        last_layer=False,
        **kwargs
    ):
        super().__init__(block, num_blocks, use_sn=use_sn, **kwargs)
        self.dropout_rate = dropout_rate
        self.last_layer = last_layer
        self.feature = None

    #
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        if not self.last_layer:
            out = nn.functional.dropout(out, p=self.dropout_rate)
        out = self.layer1(out)

        if not self.last_layer:
            out = nn.functional.dropout(out, p=self.dropout_rate)
        out = self.layer2(out)

        if not self.last_layer:
            out = nn.functional.dropout(out, p=self.dropout_rate)
        out = self.layer3(out)

        if not self.last_layer:
            out = nn.functional.dropout(out, p=self.dropout_rate)
        out = self.layer4(out)

        out = nn.functional.dropout(out, p=self.dropout_rate)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet50_dropout(dropout_rate=0.0, use_sn=False, **kwargs):
    return ResNetDropout(
        Bottleneck,
        [3, 4, 6, 3],
        dropout_rate=dropout_rate,
        use_sn=use_sn,
        **kwargs
    )


def ResNet18(use_sn, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_sn=use_sn, **kwargs)


def ResNet34(use_sn, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], use_sn=use_sn, **kwargs)


def ResNet50(use_sn, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], use_sn=use_sn, **kwargs)


def ResNet101(use_sn, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], use_sn=use_sn, **kwargs)


def ResNet152(use_sn, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], use_sn=use_sn, **kwargs)


def test():
    net = ResNet18(use_sn=True)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    net = ResNet18(use_sn=False)
    y = net(torch.randn(1, 3, 32, 32))
    summary(net, (3, 32, 32))
    print(y.size())


if __name__ == "__main__":
    test()
