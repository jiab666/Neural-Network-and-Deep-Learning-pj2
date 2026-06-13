"""
VGG models for CIFAR-10 experiments.
"""
import numpy as np
from torch import nn

from utils.nn import init_weights_


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


def build_activation(name, inplace=True):
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name == "elu":
        return nn.ELU(alpha=1.0, inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


def scale_channel(value, width_multiplier):
    scaled = int(round(value * width_multiplier))
    return max(8, scaled)


class ConfigurableVGG_A(nn.Module):
    """Configurable VGG-A backbone for CIFAR-10."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        use_batch_norm=False,
        dropout_p=0.0,
        activation_name="relu",
        width_multiplier=1.0,
        hidden_dim=None,
    ):
        super().__init__()

        channels = [scale_channel(base, width_multiplier) for base in [64, 128, 256, 512, 512]]
        hidden_dim = hidden_dim or channels[-1]

        self.output_dim = channels[-1]
        self.activation_name = activation_name
        self.width_multiplier = width_multiplier
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        self.features = nn.Sequential(
            *self._make_stage(inp_ch, channels[0], convs=1, use_batch_norm=use_batch_norm, activation_name=activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._make_stage(channels[0], channels[1], convs=1, use_batch_norm=use_batch_norm, activation_name=activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._make_stage(channels[1], channels[2], convs=2, use_batch_norm=use_batch_norm, activation_name=activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._make_stage(channels[2], channels[3], convs=2, use_batch_norm=use_batch_norm, activation_name=activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._make_stage(channels[3], channels[4], convs=2, use_batch_norm=use_batch_norm, activation_name=activation_name),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        classifier_layers = []
        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(self.output_dim, hidden_dim))
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(hidden_dim))
        classifier_layers.append(build_activation(activation_name, inplace=True))
        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(hidden_dim))
        classifier_layers.append(build_activation(activation_name, inplace=True))
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        if init_weights:
            self._init_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, convs, use_batch_norm, activation_name):
        layers = []
        current_in = in_channels
        for _ in range(convs):
            layers.append(nn.Conv2d(in_channels=current_in, out_channels=out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(build_activation(activation_name, inplace=True))
            current_in = out_channels
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, self.output_dim))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A(ConfigurableVGG_A):
    """Plain VGG-A baseline."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        activation_name="relu",
        width_multiplier=1.0,
        hidden_dim=None,
    ):
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            init_weights=init_weights,
            use_batch_norm=False,
            dropout_p=0.0,
            activation_name=activation_name,
            width_multiplier=width_multiplier,
            hidden_dim=hidden_dim,
        )


class VGG_A_BatchNorm(ConfigurableVGG_A):
    """VGG-A with BatchNorm layers."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        activation_name="relu",
        width_multiplier=1.0,
        hidden_dim=None,
    ):
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            init_weights=init_weights,
            use_batch_norm=True,
            dropout_p=0.0,
            activation_name=activation_name,
            width_multiplier=width_multiplier,
            hidden_dim=hidden_dim,
        )


class VGG_A_Dropout(ConfigurableVGG_A):
    """VGG-A with Dropout in the classifier."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        activation_name="relu",
        width_multiplier=1.0,
        hidden_dim=None,
        dropout_p=0.5,
    ):
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            init_weights=init_weights,
            use_batch_norm=False,
            dropout_p=dropout_p,
            activation_name=activation_name,
            width_multiplier=width_multiplier,
            hidden_dim=hidden_dim,
        )


class VGG_A_Light(ConfigurableVGG_A):
    """Smaller VGG-A variant kept for convenience."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            init_weights=init_weights,
            use_batch_norm=False,
            dropout_p=0.0,
            activation_name="relu",
            width_multiplier=0.25,
            hidden_dim=128,
        )


if __name__ == "__main__":
    print(get_number_of_parameters(VGG_A()))
    print(get_number_of_parameters(VGG_A_BatchNorm()))
    print(get_number_of_parameters(VGG_A_Light()))
    print(get_number_of_parameters(VGG_A_Dropout()))
