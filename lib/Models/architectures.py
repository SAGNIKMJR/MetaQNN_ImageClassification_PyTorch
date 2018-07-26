from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    
    return num_features


def get_feat_spatial_size(block, spatial_size, ncolors=3):
    x = torch.randn(1, ncolors, spatial_size, spatial_size)
    out = block(x)
    spatial_dim_x = out.size()[2]
    spatial_dim_y = out.size()[3]

    return spatial_dim_x, spatial_dim_y


# TODO: expose and include dropout parameter
class MLP(nn.Module):
    def __init__(self, num_classes, num_colors, args):
        super(MLP, self).__init__()

        self.patch_size = args.patch_size
        self.batch_norm = args.batch_norm

        self.fc1 = nn.Linear(num_colors * self.patch_size * self.patch_size, 300)
        self.bn1 = nn.BatchNorm1d(300, eps=self.batch_norm)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(300, 200)
        self.bn2 = nn.BatchNorm1d(200, eps=self.batch_norm)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100, eps=self.batch_norm)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, get_num_flat_features(x))
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, fan_in, fan_out, num_layers=2, batch_norm=1e-3, kernel_size=3,
                 padding=1, stride=1, dropout=0.0, pool=True, pool_size=2, pool_stride=2, all_conv=False):
        super(ConvBlock, self).__init__()

        self.filter_dims = (num_layers + 1) * [fan_out]
        self.filter_dims[0] = fan_in

        self.block = nn.Sequential(OrderedDict([
            ('conv_layer' + str(l+1), SingleConvLayer(l + 1, self.filter_dims[l], self.filter_dims[l + 1],
                                                      kernel_size=kernel_size, padding=padding,
                                                      stride=stride, batch_norm=batch_norm))
            for l in range(num_layers)
        ]))

        if pool:
            if all_conv:
                self.block.add_module('conv_layer_pool', nn.Conv2d(self.filter_dims[self.filter_dims.size(0)-1],
                                                                   self.filter_dims[self.filter_dims.size(0)-1],
                                                                   kernel_size=pool_size, stride=pool_stride,
                                                                   bias=False))
            else:
                self.block.add_module('mp', nn.MaxPool2d(pool_size, pool_stride))

        if not dropout == 0.0:
            self.block.add_module('dropout', nn.Dropout2d(p=dropout, inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class SingleConvLayer(nn.Module):
    def __init__(self, l, fan_in, fan_out, kernel_size=3, padding=1,
                 stride=1, batch_norm=1e-3):
        super(SingleConvLayer, self).__init__()

        self.convlayer = nn.Sequential(OrderedDict([
            ('conv' + str(l), nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                        bias=False)),
            ('bn' + str(l), nn.BatchNorm2d(fan_out, eps=batch_norm)),
            ('act' + str(l), nn.ReLU())
        ]))

    def forward(self, x):
        x = self.convlayer(x)
        return x


class ClassifierBlock(nn.Module):
    def __init__(self, fan_in, fan_out, num_classes, num_layers=2, batch_norm=1e-3, dropout=0.0):
        super(ClassifierBlock, self).__init__()

        self.filter_dims = (num_layers + 1) * [fan_out]
        self.filter_dims[0] = fan_in

        self.fc_block = nn.Sequential(OrderedDict([
            ('fc_layer' + str(l), SingleLinearLayer(l+1, self.filter_dims[l], self.filter_dims[l+1],
                                                    batch_norm=batch_norm))
            for l in range(num_layers)
        ]))

        if not dropout == 0.0:
            self.fc_block.add_module('dropout', nn.Dropout(p=dropout, inplace=True))

        self.fc_block.add_module('final_layer', nn.Linear(fan_out, num_classes))

    def forward(self, x):
        x = x.view(-1, get_num_flat_features(x))
        x = self.fc_block(x)
        return x


class SingleLinearLayer(nn.Module):
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-3):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
            ('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm)),
            ('act' + str(l), nn.ReLU())
        ]))

    def forward(self, x):
        x = self.fclayer(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes, num_colors, args):
        super(VGG, self).__init__()

        self.batch_norm = args.batch_norm
        self.all_conv = args.all_conv

        # subtract 3 classifier layers and first two static 2 layer blocks
        self.block_depth = args.vgg_depth - 7
        assert(self.block_depth % 3 == 0)
        self.layers_per_block = int(self.block_depth / 3)

        # classifier differentiation for ImageNet vs small images
        if args.patch_size > 100:
            self.num_classifier_features = 4096
        else:
            self.num_classifier_features = 512

        self.features = nn.Sequential(OrderedDict([
            ('block1', ConvBlock(num_colors, 64, batch_norm=self.batch_norm, num_layers=2, all_conv=self.all_conv)),
            ('block2', ConvBlock(64, 128, batch_norm=self.batch_norm, num_layers=2, all_conv=self.all_conv)),
            ('block3', ConvBlock(128, 256, batch_norm=self.batch_norm,
                                 num_layers=self.layers_per_block, all_conv=self.all_conv)),
            ('block4', ConvBlock(256, 512, batch_norm=self.batch_norm,
                                 num_layers=self.layers_per_block, all_conv=self.all_conv)),
            ('block5', ConvBlock(512, 512, batch_norm=self.batch_norm,
                                 num_layers=self.layers_per_block, all_conv=self.all_conv))
        ]))

        self.feat_spatial_size_x, self.feat_spatial_size_y = get_feat_spatial_size(self.features, args.patch_size,
                                                                                   ncolors=num_colors)

        self.classifier = nn.Sequential(
            ClassifierBlock(512 * self.feat_spatial_size_x * self.feat_spatial_size_y, self.num_classifier_features,
                            num_classes, num_layers=2, batch_norm=self.batch_norm, dropout=0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GFCNN(nn.Module):
    def __init__(self, num_classes, num_colors, args):
        super(GFCNN, self).__init__()

        self.batch_norm = args.batch_norm
        self.all_conv = False

        # classifier differentiation for ImageNet vs small images
        if args.patch_size > 100:
            self.num_classifier_features = 4096
        else:
            self.num_classifier_features = 512

        self.features = nn.Sequential(OrderedDict([
            ('block1', ConvBlock(num_colors, 128, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 kernel_size=8, padding=4, pool_size=4, pool_stride=2)),
            ('block2', ConvBlock(128, 192, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 kernel_size=8, padding=3, pool_size=4, pool_stride=2)),
            ('block3', ConvBlock(192, 192, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 kernel_size=5, padding=3))
        ]))

        self.feat_spatial_size_x, self.feat_spatial_size_y = get_feat_spatial_size(self.features, args.patch_size,
                                                                                   ncolors=num_colors)

        self.classifier = nn.Sequential(
            ClassifierBlock(192 * self.feat_spatial_size_x * self.feat_spatial_size_y, self.num_classifier_features,
                            num_classes, num_layers=2, batch_norm=self.batch_norm, dropout=0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Alexnet(nn.Module):
    def __init__(self, num_classes, num_colors, args):
        super(Alexnet, self).__init__()

        self.batch_norm = args.batch_norm
        self.all_conv = args.all_conv

        # AlexNet does not work for patch_sizes smaller than 96
        assert(args.patch_size >= 96)
        # classifier differentiation for ImageNet vs small images
        if args.patch_size > 100:
            self.num_classifier_features = 4096
        else:
            self.num_classifier_features = 512

        self.features = nn.Sequential(OrderedDict([
            ('block1', ConvBlock(num_colors, 64, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 kernel_size=11, stride=4, padding=2, pool_size=3, pool_stride=2)),
            ('block2', ConvBlock(64, 192, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 kernel_size=5, padding=2, pool_size=3, pool_stride=2)),
            ('block3', ConvBlock(192, 384, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 pool=False)),
            ('block4', ConvBlock(384, 256, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 pool=False)),
            ('block5', ConvBlock(256, 256, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 pool_size=3, pool_stride=2)),
            ('block6', ConvBlock(256, 256, batch_norm=self.batch_norm, num_layers=1, all_conv=self.all_conv,
                                 pool=False))
        ]))

        self.feat_spatial_size_x, self.feat_spatial_size_y = get_feat_spatial_size(self.features, args.patch_size,
                                                                                   ncolors=num_colors)

        self.classifier = nn.Sequential(
            ClassifierBlock(256 * self.feat_spatial_size_x * self.feat_spatial_size_y, self.num_classifier_features,
                            num_classes, num_layers=2, batch_norm=self.batch_norm, dropout=0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, batchnorm=1e-3):
        super(WRNBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.droprate = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WRNNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block_type, dropout=0.0, batchnorm=1e-3, stride=1):
        super(WRNNetworkBlock, self).__init__()

        self.block = nn.Sequential(OrderedDict([
            ('conv_block' + str(layer+1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                     layer == 0 and stride or 1, dropout, batchnorm=batchnorm))
            for layer in range(nb_layers)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class WideResNet(nn.Module):
    """
    Adapted from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
    """
    def __init__(self, num_classes, num_colors, args):
        super(WideResNet, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth
        self.batch_norm = args.batch_norm
        drop_rate = 0.3  # TODO: currently hardcoded!!

        self.nChannels = [16, 16 * self.widen_factor, 32 * self.widen_factor, 64 * self.widen_factor]
        assert((self.depth - 4) % 6 == 0)
        self.num_block_layers = int((self.depth - 4) / 6)

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
            ('block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1], WRNBasicBlock,
                                       dropout=drop_rate, batchnorm=self.batch_norm)),
            ('block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2], WRNBasicBlock,
                                       dropout=drop_rate, batchnorm=self.batch_norm, stride=2)),
            ('block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3], WRNBasicBlock,
                                       dropout=drop_rate, batchnorm=self.batch_norm, stride=2)),
            ('bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
            ('act1', nn.ReLU(inplace=True)),
            ('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        ]))

        self.classifier = nn.Sequential(
            ClassifierBlock(self.nChannels[3], self.nChannels[3], num_classes,
                            num_layers=0, batch_norm=self.batch_norm)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
