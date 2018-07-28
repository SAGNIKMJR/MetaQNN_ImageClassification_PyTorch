# MetaQNN_ImageClassification_PyTorch
Implementation of MetaQNN (https://arxiv.org/abs/1611.02167, https://github.com/bowenbaker/metaqnn.git) with Additions and Modifications in PyTorch for Image Classification.   
    
# Basic Search Space Specs:
 i) Minimum no. of Conv./Wrn layers   
 ii) Maximum no. of Conv./Wrn layers   
 iii) Maximum 1 FC layer (classifier not counted)   

# Additions/Modifications:
 i) Optional Greedy version of Q-learning update rule added for shorter search schedules     
```python
def __update_q_value_sequence(self, states, termination_reward):
    self.__update_q_value(states[-2], states[-1], termination_reward)
    for i in reversed(range(len(states) - 2)):
        
        # NOTE: q-learning update (set proper q-learning rate in cmdparser.py)
        self.__update_q_value(states[i], states[i+1], 0)

        # NOTE: modified update for shorter search schedules (doesn't use q-learning rate in computation)
        # self.__update_q_value(states[i], states[i+1], termination_reward)

def __update_q_value(self, start_state, to_state, reward):
    if start_state.as_tuple() not in self.qstore.q:
        self.enum.enumerate_state(start_state, self.qstore.q)
    if to_state.as_tuple() not in self.qstore.q:
        self.enum.enumerate_state(to_state, self.qstore.q)

    actions = self.qstore.q[start_state.as_tuple()]['actions']
    values = self.qstore.q[start_state.as_tuple()]['utilities']

    max_over_next_states = max(self.qstore.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0
    action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

    # NOTE: q-learning update (set proper q-learning rate in cmdparser.py)
    values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                   self.args.q_learning_rate * \
                                                   (reward + self.args.q_discount_factor *
                                                    max_over_next_states -
                                                    values[actions.index(action_between_states)])

    # NOTE: modified update for shorter search schedules (doesn't use q-learning rate in computation)
    # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
    #                                                (max(reward, values[actions.index(action_between_states)]) -
    #                                                 values[actions.index(action_between_states)])

    self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
```
 ii) Skip connections with WideResNet blocks, minimum and maximum conv layer limit and made some other search space changes for better performace    
 iii) Continuing from the previous Q-learning iteration if code crashes  while running      
 iv) Running over single or multiple GPUs    
 iv) Automatic calculation of available GPU space and skipping of architecture if it doesn't fit      
             
# NOTE:    
code for MNIST, CIFAR10 and CIFAR100; for other datasets dataloader has to be added to _lib/Datasets/datasets.py_    
# Running Search:       
Use _python 2.7_ and _torch 0.4.0_      
Look at __lib/cmdparser.py__ for the available command line options or just run 
```sh
$ python main.py --help
```
      
Finally, run __main.py__
