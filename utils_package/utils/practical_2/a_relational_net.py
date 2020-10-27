import nshot
import torch
import torch.nn as nn
import random
import itertools


class ARelationalNet(nshot.models.model.ModelBaseClass):
    def __init__(self, encoder, fc_dim=64, activation='ELU',
                 normalization='layernorm', n_blocks=4, pre_encoded=False,
                 max_comparisons=50, device='cpu', place_on_device=True):
        super(ARelationalNet, self).__init__()

        self.name = 'ARelationalNet'

        self.config = {
            'fc_dim': fc_dim,
            'activation': activation,
            'normalization': normalization,
            'n_blocks': n_blocks,
            'pre_encoded': pre_encoded,
            'max_comparisons': max_comparisons,
        }

        self.hyper_parameters = {
            'fc_dim': fc_dim,
            'activation': activation,
            'normalization': normalization,
            'n_blocks': n_blocks,
            'max_comparisons': max_comparisons
        }

        # Setup activation function
        self.activation = activation

        possible_activations = ['ELU', 'Tanh', 'Sigmoid', 'ReLU', 'SELU']
        if isinstance(self.activation, str):
            if self.activation not in possible_activations:
                raise ValueError('invalid activation function')

        # Setup normalization
        self.normalization = normalization
        if isinstance(self.normalization, str):
            normalizations = [
                'weightnorm',
                'batchnorm',
                'layernorm',
                'groupnorm'
            ]

            if self.normalization not in normalizations:
                raise ValueError('invalid normalization function')

        self.device = device
        self.place_on_device = place_on_device
        self.max_comparisons = max_comparisons
        self.fc_dim = fc_dim
        self.n_blocks = n_blocks
        self.pre_encoded = pre_encoded
        self.encoder = encoder

        self.relational = Relational(
            self.encoder.output_shape,
            fc_dim,
            activation,
            normalization,
            n_blocks
        )

        self.comparison = Comparison(
            image_features=self.relational.image_features,
            class_features=fc_dim,
            activation=activation,
            normalization=normalization,
            n_blocks=n_blocks
        )

        self.classification = Classification(
            image_features=self.comparison.image_features,
            class_features=fc_dim,
            activation=activation,
            normalization=normalization,
            n_blocks=n_blocks
        )

        if self.place_on_device:
            self.assign_devices()

    def assign_devices(self):
        """Assigns the model and encoder to their devices.
        """
        self.to(self.device)

        if self.encoder.device is None:
            self.encoder.device = self.device

        self.encoder.to(self.encoder.device)

    def summary(self):
        """Produces a dictionary with summary items for the RCNet model.

        :return: Model summary items
        :rtype: dict
        """
        summary_dictionary = {
            'Encoder': self.encoder.__class__.__name__,
            'fc_dim': self.fc_dim,
            'Activation': self.activation,
            'Normalization': self.normalization,
            'n_blocks': self.n_blocks
            }

        return summary_dictionary

    def forward(self, episode):
        support = episode['support']
        query = episode['query']
        support_labels = episode['support_labels']

        del episode

        class_count = max([int(a[1:]) for a in support_labels if 'S' in a]) + 1

        class_iterator = range(class_count)
        support_labels = support_labels + ['Q'] * len(query)

        datapoints = self.encoder(support + query)

        if self.place_on_device:
            datapoints = datapoints.to(self.device)

        index = ['Q' == a for a in support_labels]
        index = [i for i, a in enumerate(index) if a]
        query = datapoints[index, ...]

        # Send each positive support set through the relational network
        positives = []
        for i in class_iterator:
            # NOTE: If you are running 5-shot 5-way then the tensor object
            # three lines down should be a tensor of shape 5 by feature
            # dimension.
            index = ['S' + str(i) == a for a in support_labels]
            index = [i for i, a in enumerate(index) if a]
            tensor = datapoints[index, ...]

            comparisons = itertools.permutations(range(tensor.shape[0]), 2)
            comparisons = list(comparisons)

            pairs = int(min(self.max_comparisons, len(comparisons)))

            try:
                left, right = zip(*random.sample(comparisons, pairs))
            except:
                left, right = 0, 0

            # NOTE: The tensor[left, :] object should have shape pairs by
            # feature dimension. The same is true for tensor[right, :].
            output = self.relational(
                tensor[left, :],
                tensor[right, :]
            )

            output = output.mean(0)  # shape: 1 by new feature dimension
            output = output.unsqueeze(0).expand(query.shape[0], -1)
            positives.append(output)

        results = []

        # Send each query image through the comparison stage
        for i in class_iterator:
            results.append(
                self.comparison(
                    query,
                    positives[i]
                )
            )

        # Send entire episode through the final classification stage
        class_features = torch.stack(results, 1)

        logits = self.classification(query, class_features)

        return logits


class Stage(nn.Module):
    def __init__(self):
        """baseclass for the various parts of the RCNet architecture.
        """
        super(Stage, self).__init__()

    def forward(self):
        raise NotImplementedError

    def make_block(self, n_features, fc_dim, activation):
        """Model construction of layers, normalization, and activation

        :param n_features: Number of input features
        :type n_features: int
        :param fc_dim: Output dimension from previous stage
        :type fc_dim: int
        :param activation: A torch activation function or (str) one of 'ELU',
            'Tanh', 'Sigmoid', or 'ReLU'
        :type activation: str
        :return: A torch sequential container of normalized linear layers
        :rtype: torch.nn.Module
        """
        if isinstance(activation, str):
            activation_functions = {
                'ELU': nn.ELU,
                'Tanh': nn.Tanh,
                'Sigmoid': nn.Sigmoid,
                'ReLU': nn.ReLU,
                'SELU': nn.SELU
            }

            try:
                activation = activation_functions[activation]
            except KeyError:
                raise ValueError('invalid activation function')

        if isinstance(self.normalization, str):
            normalizations = [
                'weightnorm',
                'batchnorm',
                'layernorm',
                'groupnorm'
            ]

            if self.normalization not in normalizations:
                raise ValueError('invalid normalization function')

            if self.normalization == 'weightnorm':
                layers = [
                    nn.utils.weight_norm(
                        nn.Linear(n_features, fc_dim)
                    ),
                    activation()
                ]
            elif self.normalization == 'batchnorm':
                layers = [
                    nn.Linear(n_features, fc_dim),
                    nn.BatchNorm1d(fc_dim),
                    activation()
                ]
            elif self.normalization == 'layernorm':
                layers = [
                    nn.Linear(n_features, fc_dim),
                    nn.LayerNorm(fc_dim),
                    activation()
                ]
            elif self.normalization == 'groupnorm':
                layers = [
                    nn.Linear(n_features, fc_dim),
                    nn.GroupNorm(num_groups=2, num_channels=fc_dim),
                    activation()
                ]
        else:
            if self.normalization is None:
                layers = [
                    nn.Linear(n_features, fc_dim),
                    activation()
                ]
            else:
                layers = [
                    nn.Linear(n_features, fc_dim),
                    self.normalization(),
                    activation()
                ]

        block = nn.Sequential(*layers)

        return block

    def init(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, a=1.0)
            nn.init.constant_(layer.bias, val=0.1)


class Relational(Stage):
    def __init__(self, image_features, fc_dim=128, activation=nn.ELU,
                 normalization=None, n_blocks=4):

        super(Relational, self).__init__()
        self.fc_dim = fc_dim
        self.activation = activation
        self.normalization = normalization
        self.image_features = image_features

        self.blocks = []
        for i in range(n_blocks):
            if i == 0:
                n_features = image_features * 2
            else:
                n_features = self.fc_dim

            self.blocks.append(
                self.make_block(n_features, self.fc_dim, activation)
            )

        self.blocks = nn.ModuleList(self.blocks)
        self.apply(self.init)

    def forward(self, left, right):

        if len(left.shape) == 1:
            left = left.unsqueeze(0)
            right = right.unsqueeze(0)

        x = torch.cat([left, right], dim=1)

        x1 = self.blocks[0](x)

        if len(self.blocks) >= 2:
            x = self.blocks[1](x1)

        if len(self.blocks) >= 3:
            for i in range(2, len(self.blocks)):
                x = self.blocks[i](x)

        if len(self.blocks) >= 3:
            x = torch.add(x, x1)

        return x


class Comparison(Stage):
    def __init__(self, image_features, class_features, activation=nn.ELU,
                 normalization=None, n_blocks=4):

        super(Comparison, self).__init__()
        self.fc_dim = class_features
        self.activation = activation
        self.normalization = normalization
        self.image_features = image_features

        total_features = image_features + class_features

        self.blocks = []
        for i in range(n_blocks):
            if i == 0:
                n_features = total_features
            else:
                n_features = self.fc_dim

            self.blocks.append(
                self.make_block(
                    n_features,
                    self.fc_dim,
                    activation
                )
            )

        self.blocks = nn.ModuleList(self.blocks)

        self.apply(self.init)

    def forward(self, image, class_vector):
        x = torch.cat([image, class_vector], dim=1)

        x1 = self.blocks[0](x)

        if len(self.blocks) >= 2:
            x = self.blocks[1](x1)

        if len(self.blocks) >= 3:
            for i in range(2, len(self.blocks)):
                x = self.blocks[i](x)

        if len(self.blocks) >= 3:
            x = torch.add(x, x1)

        return x


class Classification(Stage):
    def __init__(self, image_features, class_features, activation=nn.ELU,
                 normalization=None, n_blocks=3):

        super(Classification, self).__init__()
        self.fc_dim        = class_features
        self.activation    = activation
        self.normalization = normalization

        none_class = torch.zeros(class_features, requires_grad=True).float()
        self.none_class = nn.Parameter(data=none_class)

        self.blocks = []
        for i in range(n_blocks):
            if i == 0:
                n_features = image_features + class_features
            else:
                n_features = self.fc_dim
            self.blocks.append(self.make_block(n_features, self.fc_dim,
                               activation))
        self.blocks = nn.ModuleList(self.blocks)
        self.final  = nn.Linear(self.fc_dim, 1)

        self.apply(self.init)

    def forward(self, image, class_vectors):
        none_class = self.none_class.unsqueeze(0).unsqueeze(1)
        none_class = none_class.expand(class_vectors.size(0), -1, -1)

        class_vectors = torch.cat([class_vectors, none_class], dim=1)

        image = image.unsqueeze(1).expand(-1, class_vectors.size(1), -1)
        x = torch.cat([image, class_vectors], dim=2)

        logits = []

        for ix in range(class_vectors.size(1)):
            x1_ = self.blocks[0](x[:, ix, :])
            if len(self.blocks) >= 2:
                x_ = self.blocks[1](x1_)

            if len(self.blocks) >= 3:
                for jx in range(2, len(self.blocks)):
                    x_ = self.blocks[jx](x_)

            if len(self.blocks) >= 3:
                x_ = torch.add(x_, x1_)

            x_ = self.final(x_)

            logits.append(x_)

        logits = torch.cat(logits, dim=1)

        return logits