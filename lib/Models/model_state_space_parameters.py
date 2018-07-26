# Transition Options
possible_conv_depths = [32, 64, 128, 256]       # Choices for number of filters in a convolutional layer\
"""
    NOTE: 
    conv sizes of 3,5 have stride = 1; conv sizes of 7,9,11 have stride = 2
"""
possible_conv_sizes = [3,5,7]              # Choices for conv kernel size (square)
possible_spp_sizes = [3,4]
"""
    NOTE: 
    No pool
"""
# possible_pool_sizes = [5, 3, 2]               # Choices for pool kernel size (square)
# possible_pool_strides = [3, 2, 2]             # Choices for pool stride (symmetric)
possible_fc_sizes = [i for i in [128, 64, 32]]  # Possible number of neurons in a fully connected layer

allow_initial_pooling = False                   # Allow pooling as the first layer
allow_consecutive_pooling = False               # Allow a pooling layer to follow a pooling layer

# conv_padding = 'SAME'                         # set to 'SAME' to pad convolutions so input and output dimension are the same
conv_padding = 'VALID'                          # set to 'VALID' not to pad convolutions


"""
    Q-Learning Hyper parameters
"""

# Epsilon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
# NOTEL schedule for search with fixed batch sizes
# epsilon_schedule = [[1.0, 100],
#                     [0.9, 10],
#                     [0.8, 10],
#                     [0.7, 10],
#                     [0.6, 10],
#                     [0.5, 15],
#                     [0.4, 15],
#                     [0.3, 15],
#                     [0.2, 20],
#                     [0.1, 20],
#                     [0.0, 20]]

# NOTE: schedule for search with variable batch sizes
epsilon_schedule = [[1.0, 250],
                    [0.9, 20],
                    [0.8, 20],
                    [0.7, 20],
                    [0.6, 20],
                    [0.5, 20],
                    [0.4, 30],
                    [0.3, 40],
                    [0.2, 40],
                    [0.1, 40],
                    [0.0, 40]]



replay_number = 40                              # Number trajectories to sample for replay at each iteration

# Set up the representation size buckets (see paper Appendix Section B)
def image_size_bucket(image_size):
    if image_size > 7:                          # image size between 8 and infinity
        return 8
    elif image_size > 3:                        # image size between 4 and 7
        return 4
    else:                                       # image size between 1 and 3
        return 1  
