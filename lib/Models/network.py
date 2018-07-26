import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from lib.Utility import FeatureOperations as FO

class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, batchnorm=1e-3):
        super(WRNBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.droprate = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        """ 
            NOTE: batch norm in commented lines
        """
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            # x = F.relu(x)

        else:
            out = self.relu1(self.bn1(x))
            # out = F.relu(x)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class net(nn.Module):

    def __init__(self, state_list, state_space_parameters, num_classes, net_input, bn_val, do_drop):
        super(net, self).__init__()
        self.state_list = state_list
        self.state_space_parameters = state_space_parameters
        self.batch_size = net_input.size(0)
        self.num_colors = net_input.size(1)
        self.image_size = net_input.size(2)
        self.num_classes = num_classes
        self.bn_value = bn_val
        self.do_drop = do_drop
        self.gpu_usage = 32 * self.batch_size * self.num_colors * self.image_size * self.image_size
        feature_list = []	
        classifier_list = []												
        wrn_bb_no = conv_no = pool_no = fc_no = relu_no = drop_no = bn_no = 0 
        feature = 1
        out_channel = self.num_colors
        no_feature = self.num_colors*((self.image_size)**2)
        print('=' * 80)
        for state_no, state in enumerate(self.state_list):
            if state_no == len(self.state_list)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'wrn':
                    wrn_bb_no += 1 #conv_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    last_image_size = state.image_size
                    # TODO: fix padding, will work for stride = 1 only
                    feature_list.append(('wrn_bb' + str(wrn_bb_no), WRNBasicBlock(in_channel, out_channel,
                                        stride = state.stride, dropout = 0, batchnorm = self.bn_value)))
                    self.gpu_usage += 32*(3*3*in_channel*out_channel + 3*3*out_channel*out_channel + int(in_channel!=out_channel)*in_channel*out_channel)
                    self.gpu_usage += 32*self.batch_size*state.image_size*state.image_size*state.filter_depth*(2 + int(in_channel!=out_channel))

                    # feature_list.append(('dropout' + str(wrn_bb_no), nn.Dropout2d(p = self.do_drop)))   
                elif state.layer_type == 'conv':
                    conv_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    last_image_size = state.image_size
                    # TODO: include option for 'SAME'
                    # TODO: fix padding, will work for stride = 1 only
                    feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel,
                                        state.filter_size, stride = state.stride, padding = 0, bias = False)))   
                    """
                        NOTE:
                        uncomment to include batch norm
                    """                 
                    bn_no += 1
                    feature_list.append(('batchnorm' + str(bn_no), nn.BatchNorm2d(num_features=out_channel, eps=self.bn_value)))
                    relu_no += 1
                    feature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
                    # feature_list.append(('dropout' + str(conv_no), nn.Dropout2d(p = do_drop)))
                    self.gpu_usage += 32*(state.image_size * state.image_size * state.filter_depth * self.batch_size \
                                        + in_channel * out_channel * state.filter_size * state.filter_size)
                elif state.layer_type == 'spp':
                    # TODO: dirty hack for calculating spp feature size 
                    temp = torch.randn(self.batch_size, out_channel, last_image_size, last_image_size)
                    no_feature = FO.spatial_pyramid_pooling(temp, state.filter_size).size(1)#out_channel * int(state.filter_size*(state.filter_size + 1)*(2*state.filter_size + 1)/6.)
                    self.gpu_usage += 32 * no_feature * self.batch_size
                    self.spp_filter_size = state.filter_size 
            else:
                if state.layer_type == 'fc':
                    fc_no += 1
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    self.gpu_usage += no_feature
                    classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature, bias = False)))
                    """
                        NOTE:
                        uncomment to include batch norm
                    """      
                    classifier_list.append(('batchnorm_fc' + str(fc_no), nn.BatchNorm1d(num_features=no_feature, eps=self.bn_value)))
                    classifier_list.append(('relu_fc' + str(fc_no), nn.ReLU(inplace=True)))
                    # classifier_list.append(('dropout' + str(fc_no), nn.Dropout(p = do_drop)))
                    self.gpu_usage += 32 * (no_feature * self.batch_size + in_feature * no_feature)

        self.features_list = nn.Sequential(collections.OrderedDict(feature_list))

        classifier_list.append(('dropout' + str(no_feature), nn.Dropout(p = do_drop)))
        classifier_list.append(('fc' + str(fc_no+1), nn.Linear(no_feature, self.num_classes, bias = False)))
        self.classifiers_list = nn.Sequential(collections.OrderedDict(classifier_list))
        self.gpu_usage += 32 * (self.num_classes * self.batch_size + no_feature * self.num_classes)
        self.gpu_usage /= (8.*1024*1024*1024)
    def forward(self, x):
        x = FO.spatial_pyramid_pooling(self.features_list(x), self.spp_filter_size)
        x = x.view(x.size(0), -1)
        x = self.classifiers_list(x)
        x = F.sigmoid(x)
        return x