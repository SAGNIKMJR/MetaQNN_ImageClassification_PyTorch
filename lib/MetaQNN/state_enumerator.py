import math


class State:
    def __init__(self,
                 layer_type=None,        # String -- conv, pool, fc, softmax
                 layer_depth=None,       # Current depth of network
                 filter_depth=None,      # Used for conv, 0 when not conv
                 filter_size=None,       # Used for conv and pool, 0 otherwise
                 stride=None,            # Used for conv and pool, 0 otherwise
                 image_size=None,        # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 fc_size=None,           # Used for fc and softmax -- number of neurons in layer
                 terminate=None,
                 state_list=None):       # can be constructed from a list instead, list takes precedent
        if not state_list:
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.filter_depth = filter_depth
            self.filter_size = filter_size
            self.stride = stride
            self.image_size = image_size
            self.fc_size = fc_size
            self.terminate = terminate
        else:
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.filter_depth = state_list[2]
            self.filter_size = state_list[3]
            self.stride = state_list[4]
            self.image_size = state_list[5]
            self.fc_size = state_list[6]
            self.terminate = state_list[7]

    def as_tuple(self):
        return (self.layer_type, 
                self.layer_depth, 
                self.filter_depth, 
                self.filter_size, 
                self.stride, 
                self.image_size,
                self.fc_size,
                self.terminate)

    def as_list(self):
        return list(self.as_tuple())

    def copy(self):
        return State(self.layer_type, 
                     self.layer_depth, 
                     self.filter_depth, 
                     self.filter_size, 
                     self.stride, 
                     self.image_size,
                     self.fc_size,
                     self.terminate)


class StateEnumerator:
    def __init__(self, state_space_parameters, args):
        # Limits
        self.ssp = state_space_parameters
        self.args = args
        self.conv_layer_min_limit = args.conv_layer_min_limit
        self.conv_layer_max_limit = args.conv_layer_max_limit
        # TODO: hard-coded now to early stopping criterion, should be a user input
        self.init_utility = args.init_utility

    def enumerate_state(self, state, q_values):
        """Defines all state transitions, populates q_values where actions are valid

        Legal Transitions:
           conv         -> conv, pool                   (IF state.layer_depth < layer_limit)
           conv         -> fc                           (If state.layer_depth < layer_limit)
           conv         -> softmax, gap                 (Always)

           pool          -> conv,                       (If state.layer_depth < layer_limit)
           pool          -> fc,                         (If state.layer_depth < layer_limit)
           pool          -> softmax, gap                (Always)

           fc           -> fc                           (If state.layer_depth < layer_limit AND state.filter_depth < 3)
           fc           -> softmax                      (Always)

           gap          -> softmax                      (Always)

        Updates: q_values and returns q_values
        """
        actions = []

        if state.terminate == 0:

            # If we are at the layer limit, we can only go to softmax
            if state.layer_type == 'fc' or state.layer_type == 'spp':
              actions += [State(layer_type=state.layer_type,
                                layer_depth=state.layer_depth + 1,
                                filter_depth=state.filter_depth,
                                filter_size=state.filter_size,
                                stride=state.stride,
                                image_size=state.image_size,
                                fc_size=state.fc_size,
                                terminate=1)]

            if state.layer_type == 'spp':
                for fc_size in self._possible_fc_size(state):
                    actions += [State(layer_type='fc',
                                      layer_depth=state.layer_depth + 1,
                                      filter_depth=0,
                                      filter_size=0,
                                      stride=0,
                                      image_size=0,
                                      fc_size=fc_size,
                                      terminate=0)]
            # FC -> FC 
            if state.layer_type == 'fc'and state.filter_depth < self.args.max_fc - 1:
                for fc_size in self._possible_fc_size(state):
                    actions += [State(layer_type='fc',
                                      layer_depth=state.layer_depth + 1,
                                      filter_depth=state.filter_depth + 1,
                                      filter_size=0,
                                      stride=0,
                                      image_size=0,
                                      fc_size=fc_size,
                                      terminate=0)]

            if state.layer_depth < self.conv_layer_min_limit:
                if state.layer_type in ['start', 'conv', 'wrn']:
                    # Conv/WRN -> Conv 
                    # Conv states -- iterate through all possible depths, filter sizes
                    for depth in self.ssp.possible_conv_depths:
                        for filt in self._possible_conv_sizes(state.image_size):
                            # TODO: hard-coded check with state.image_size
                            if self.ssp.conv_padding == 'SAME' and state.image_size >= min(self.ssp.possible_spp_sizes): #13:
                                actions += [State(layer_type='conv',
                                                  layer_depth=state.layer_depth + 1,
                                                  filter_depth=depth,
                                                  filter_size=filt,
                                                  # TODO: harcoded-stride for different filter size
                                                  stride=1 if filt <=5 else 2,
                                                  image_size=state.image_size, 
                                                  fc_size=0,
                                                  terminate=0)]   
                            # TODO: hard-coded check with state.image_size 
                            elif self.ssp.conv_padding == 'VALID' and self._calc_new_image_size(state.image_size, filt) >= min(self.ssp.possible_spp_sizes): #13:
                                actions += [State(layer_type='conv',
                                                  layer_depth=state.layer_depth + 1,
                                                  filter_depth=depth,
                                                  filter_size=filt,
                                                  # TODO: harcoded-stride for different filter size
                                                  stride=1 if filt <=5 else 2,
                                                  image_size=self._calc_new_image_size(state.image_size, filt),
                                                  fc_size=0,
                                                  terminate=0)]                                    

                    # Conv/WRN -> WRN
                    # WRNBasicBlock, same starting initial conditions as conv
                    # TODO: stride fixed to 1, filter size fixed to 3
                    for depth in self.ssp.possible_conv_depths:
                        # TODO: hardcoded filter size of 3 again for possible wrn block
                        if state.image_size > 3:
                            actions += [State(layer_type='wrn',
                                              layer_depth=state.layer_depth + 2,
                                              filter_depth=depth,
                                              filter_size=3,
                                              stride=1,
                                              image_size=state.image_size,
                                              fc_size=0,
                                              terminate=0)]

            elif state.layer_depth >= self.conv_layer_min_limit and state.layer_depth < self.conv_layer_max_limit:
                if state.layer_type in ['conv', 'wrn']:
                    # Conv/WRN -> Conv-- iterate through all possible depths, filter sizes
                    for depth in self.ssp.possible_conv_depths:
                        for filt in self._possible_conv_sizes(state.image_size):
                            # TODO: hard-coded check with state.image_size
                            if self.ssp.conv_padding == 'SAME' and state.image_size >= min(self.ssp.possible_spp_sizes): #13:
                                actions += [State(layer_type='conv',
                                                  layer_depth=state.layer_depth + 1,
                                                  filter_depth=depth,
                                                  filter_size=filt,
                                                  # TODO: harcoded-stride for different filter size
                                                  stride= 1 if filt <=5 else 2,
                                                  image_size=state.image_size,
                                                  fc_size=0,
                                                  terminate=0)]
                            # TODO: hard-coded check with state.image_size
                            elif self.ssp.conv_padding == 'VALID' and self._calc_new_image_size(state.image_size, filt) >= min(self.ssp.possible_spp_sizes): #13:
                                actions += [State(layer_type='conv',
                                                  layer_depth=state.layer_depth + 1,
                                                  filter_depth=depth,
                                                  filter_size=filt,
                                                  # TODO: harcoded-stride for different filter size
                                                  stride= 1 if filt <=5 else 2,
                                                  image_size=self._calc_new_image_size(state.image_size, filt),
                                                  fc_size=0,
                                                  terminate=0)]                                

                    # Conv/WRN -> WRN 
                    # WRNBasicBlock, same starting initial conditions as conv
                    # TODO: stride fixed to 1, filter size fixed to 3
                    for depth in self.ssp.possible_conv_depths:
                        # TODO: hardcoded filter size of 3 again for possible wrn block
                        if state.image_size > 3:
                            actions += [State(layer_type='wrn',
                                              layer_depth=state.layer_depth + 2,
                                              filter_depth=depth,
                                              filter_size=3,
                                              stride=1,
                                              image_size=state.image_size,
                                              fc_size=0,
                                              terminate=0)]

                    # Conv/WRN -> SPP
                    for spp_filt in self.ssp.possible_spp_sizes:
                        if state.image_size >= spp_filt:
                            actions += [State(layer_type='spp',
                                              layer_depth=state.layer_depth + 1,
                                              filter_depth=state.filter_depth,
                                              filter_size=spp_filt,
                                              stride=0,
                                              image_size=int(spp_filt*(spp_filt + 1)*(2 * spp_filt + 1)/6.),
                                              fc_size=0,
                                              terminate=0)]   

            elif state.layer_depth >= self.conv_layer_max_limit:
                # Conv/WRN -> SPP
                if state.layer_type in ['conv', 'wrn']:
                    for spp_filt in self.ssp.possible_spp_sizes:
                        if state.image_size >= spp_filt:
                            actions += [State(layer_type='spp',
                                              layer_depth=state.layer_depth + 1,
                                              filter_depth=state.filter_depth,
                                              filter_size=spp_filt,
                                              stride=0,
                                              image_size=int(spp_filt*(spp_filt + 1)*(2 * spp_filt + 1)/6.),
                                              fc_size=0,
                                              terminate=0)]                      


        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {'actions': [to_state.as_tuple() for to_state in actions],
                                      'utilities': [self.init_utility for i in range(len(actions))]}
        # q_values_bucketed[state.as_tuple()] = {'actions': [self.bucket_state_tuple(to_state.as_tuple()) for to_state in actions],\
        #                                            'utilities': [self.init_utility for i in range(len(actions))]}

        return q_values        

    def transition_to_action(self, start_state, to_state):
        action = to_state.copy()
        # if to_state.layer_type not in ['fc', 'gap']: #if to_state.layer_type in ['fc', 'gap']:
        #     action.image_size = start_state.image_size
        return action

    def state_action_transition(self, start_state, action):
        """
            start_state: Should be the actual start_state, not a bucketed state
            action: valid action

            returns: next state, not bucketed
        """
        to_state = action.copy()
        return to_state

    def bucket_state_tuple(self, state):
        bucketed_state = State(state_list=state).copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
        return bucketed_state.as_tuple()

    def bucket_state(self, state):
        bucketed_state = state.copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
        return bucketed_state

    def _calc_new_image_size(self, image_size, filter_size):
        """
        Returns new image size given previous image size and filter parameters
        """
        # TODO: hardcoded stride setting for different filter size
        # TODO: padding hard-coded to 0
        if filter_size <= 5:
            new_size = int(math.floor((image_size - filter_size) / 1 + 1))
        else:
            new_size = int(math.floor((image_size - filter_size) / 2 + 1))
        return new_size

    def _calc_new_image_size_pool(self, image_size, pool_size, stride):
        new_size = int(math.floor(float(image_size - pool_size + 0) / float(stride) + 1))
        return new_size

    # TODO: possible conv size takes into account only the image size and not the padding
    def _possible_conv_sizes(self, image_size):
        return [conv for conv in self.ssp.possible_conv_sizes if conv < image_size]

    def _possible_pool_sizes(self, image_size):
        return [pool for pool in self.ssp.possible_pool_sizes if pool < image_size]

    def _possible_pool_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_pool_strides if stride <= filter_size]

    def _possible_fc_size(self, state):
        """
        Return a list of possible FC sizes given the current state
        """
        if state.layer_type == 'fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes

    def __allow_fully_connected(self, image_size, min_size):
        # TODO: think about allowing fully-connected layers in between convolutions
        return image_size <= min_size