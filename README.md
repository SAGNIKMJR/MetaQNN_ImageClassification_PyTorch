# MetaQNN_ImageClassification_PyTorch
Implementation of MetaQNN (https://arxiv.org/abs/1611.02167, https://github.com/bowenbaker/metaqnn.git) with Additions and Modifications in PyTorch for Image Classification.

# Additions/Modifications:
 i) Optional Greedy version of Q-learning update rule added for shorter search schedules     
```python
def __update_q_value_sequence(self, states, termination_reward):
    self.__update_q_value(states[-2], states[-1], termination_reward)
    for i in reversed(range(len(states) - 2)):
        
        # TODO: q-learning update (set proper learning rate)
        self.__update_q_value(states[i], states[i+1], 0)

        # TODO: new update (can't be used since not q-learning anymore)
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

    # TODO: q-learning update (set proper learning rate)
    values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                   self.args.q_learning_rate * \
                                                   (reward + self.args.q_discount_factor *
                                                    max_over_next_states -
                                                    values[actions.index(action_between_states)])

    # TODO: new update (can't be used since not q-learning anymore)
    # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
    #                                                self.args.q_learning_rate * \
    #                                                (max(reward, values[actions.index(action_between_states)]) -
    #                                                 values[actions.index(action_between_states)])

    self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

```
