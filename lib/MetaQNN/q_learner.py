import os
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from lib.MetaQNN.cnn import parse as cnn_parse
import state_enumerator as se
from state_string_utils import StateStringUtils
from lib.Models.network import net
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Training.learning_rate_scheduling import LearningRateScheduler
from lib.Utility.pytorch_modelsize import SizeEstimator
from lib.Utility.utils import GPUMem

class QValues:
    def __init__(self):
        self.q = {}

    def save_to_csv(self, q_csv_path):
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type': start_layer_type,
                              'start_layer_depth': start_layer_depth,
                              'start_filter_depth': start_filter_depth,
                              'start_filter_size': start_filter_size,
                              'start_stride': start_stride,
                              'start_image_size': start_image_size,
                              'start_fc_size': start_fc_size,
                              'start_terminate': start_terminate,
                              'end_layer_type': end_layer_type,
                              'end_layer_depth': end_layer_depth,
                              'end_filter_depth': end_filter_depth,
                              'end_filter_size': end_filter_size,
                              'end_stride': end_stride,
                              'end_image_size': end_image_size,
                              'end_fc_size': end_fc_size,
                              'end_terminate': end_terminate,
                              'utility': utility})
        q_csv.to_csv(q_csv_path, index=False)

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                              'start_layer_depth',
                                              'start_filter_depth',
                                              'start_filter_size',
                                              'start_stride',
                                              'start_image_size',
                                              'start_fc_size',
                                              'start_terminate',
                                              'end_layer_type',
                                              'end_layer_depth',
                                              'end_filter_depth',
                                              'end_filter_size',
                                              'end_stride',
                                              'end_image_size',
                                              'end_fc_size',
                                              'end_terminate',
                                              'utility']]):
            start_state = se.State(layer_type = row[0],
                                   layer_depth = row[1],
                                   filter_depth = row[2],
                                   filter_size = row[3],
                                   stride = row[4],
                                   image_size = row[5],
                                   fc_size = row[6],
                                   terminate = row[7]).as_tuple()
            end_state = se.State(layer_type = row[8],
                                 layer_depth = row[9],
                                 filter_depth = row[10],
                                 filter_size = row[11],
                                 stride = row[12],
                                 image_size = row[13],
                                 fc_size = row[14],
                                 terminate = row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

class QLearner:
    def __init__(self,
                 state_space_parameters, 
                 epsilon,
                 WeightInitializer=None,
                 device=None,
                 args=None,
                 save_path=None,
                 state=None,
                 qstore=None,
                 replaydict = None,
                 replay_dictionary = pd.DataFrame(columns=['net',
                                                           'spp_size',
                                                           'reward',
                                                           'epsilon',
                                                           'train_flag'])):
        self.state_list = []
        self.state_space_parameters = state_space_parameters
        self.args = args
        self.enum = se.StateEnumerator(state_space_parameters, args)
        self.stringutils = StateStringUtils(state_space_parameters, args)
        self.state = se.State('start', 0, 1, 0, 0, args.patch_size, 0, 0) if not state else state
        self.qstore = QValues() 
        if  type(qstore) is not type(None):
            self.qstore.load_q_values(qstore)
            self.replay_dictionary = pd.read_csv(replaydict, index_col=0)
        else:
            self.replay_dictionary = replay_dictionary
        self.epsilon = epsilon
        self.WeightInitializer = WeightInitializer
        self.device = device
        self.gpu_mem_0 = GPUMem(torch.device('cuda') == self.device)
        self.save_path = save_path
        # TODO: hard-coded arc no. to resume from if epsilon < 1
        self.count = args.continue_ite - 1 #137 (hard-coded no. for epsilon < 1)

    def generate_net(self, epsilon=None, dataset=None):
        if epsilon != None:
            self.epsilon = epsilon
        self.__reset_for_new_walk()
        state_list = self.__run_agent()

        net_string = self.stringutils.state_list_to_string(state_list, num_classes=len(dataset.val_loader.dataset.class_to_idx))

        train_flag = True
        if net_string in self.replay_dictionary['net'].values:
            spp_size = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['spp_size'].values[0]
            hard_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['reward'].values[0]
            train_flag = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['train_flag'].values[0]

            self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string, spp_size, hard_best_val,
                                                                                  self.epsilon, train_flag]],
                                                                                columns=['net', 
                                                                                         'spp_size',
                                                                                         'reward',
                                                                                         'epsilon',
                                                                                         'train_flag']),
                                                                   ignore_index=True)
            self.count += 1
            self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
            self.sample_replay_for_update()
            self.qstore.save_to_csv(os.path.join(self.save_path,'qVal' + str(self.count) + '.csv'))
        else:
            spp_size, hard_best_val, train_flag = self.__train_val_net(state_list, self.state_space_parameters, dataset)
            flag_net_string_present = False
            while spp_size is None:
                print('=' * 80)
                print("arc failed mem check..sampling again!")
                print('=' * 80)
                self.__reset_for_new_walk()
                state_list = self.__run_agent()
                net_string = self.stringutils.state_list_to_string(state_list, num_classes=len(dataset.val_loader.dataset.class_to_idx))
                if net_string in self.replay_dictionary['net'].values:
                    spp_size = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['spp_size'].values[0]
                    hard_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['reward'].values[0]
                    train_flag = self.replay_dictionary[self.replay_dictionary['net'] == net_string]['train_flag'].values[0]

                    self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string, spp_size, hard_best_val,
                                                                                          self.epsilon, train_flag]],
                                                                                        columns=['net', 
                                                                                                 'spp_size',
                                                                                                 'reward',
                                                                                                 'epsilon']),
                                                                           ignore_index=True)
                    self.count += 1
                    self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                    self.sample_replay_for_update()
                    self.qstore.save_to_csv(os.path.join(self.save_path,'qVal' + str(self.count) + '.csv'))
                    flag_net_string_present = True
                    break
                spp_size, hard_best_val, train_flag = \
                                                self.__train_val_net(state_list, self.state_space_parameters, dataset)
        
            if flag_net_string_present == False:

                self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string, spp_size, hard_best_val,
                                                                                      self.epsilon, train_flag]],
                                                                                    columns=['net', 
                                                                                             'spp_size',
                                                                                             'reward',
                                                                                             'epsilon',
                                                                                             'train_flag']),
                                                                       ignore_index=True)
                self.count += 1
                self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                self.sample_replay_for_update()
                self.qstore.save_to_csv(os.path.join(self.save_path,'qVal' + str(self.count) + '.csv'))

        # if train_flag == True:
        print('Reward:{}'.format(hard_best_val))

    def __train_val_net(self, state_list, state_space_parameters, dataset):
        best_prec = 0.
        num_classes = len(dataset.val_loader.dataset.class_to_idx)
        net_input, _ = next(iter(dataset.val_loader))

        model = net(state_list, state_space_parameters, num_classes, net_input, self.args.batch_norm, self.args.drop_out_drop)

        print(model)
        print('-' * 80)
        print('SPP levels: {}'.format(model.spp_filter_size))
        print('-' * 80)
        print ('Estimated total gpu usage of model: {gpu_usage:.4f} GB'.format(gpu_usage = model.gpu_usage))
        model_activations_gpu = model.gpu_usage
        cudnn.benchmark = True
        self.WeightInitializer.init_model(model)
        model = model.to(self.device)
        print('available:{}'.format((self.gpu_mem_0.total_mem - self.gpu_mem_0.total_mem*self.gpu_mem_0.get_mem_util())/1024.))
        print('required per gpu with buffer: {}'.format((3./float(self.args.no_gpus)*model_activations_gpu) + 1))
        print('-' * 80)
        if ((self.gpu_mem_0.total_mem - self.gpu_mem_0.total_mem*self.gpu_mem_0.get_mem_util())/1024.) < ((3./float(self.args.no_gpus)*model_activations_gpu) + 1): 
            del model
            return [None] * 12
        if int(self.args.no_gpus)>1:
            model = torch.nn.DataParallel(model)
        criterion = nn.BCELoss(size_average = True).to(self.device)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.args.learning_rate,
                               momentum=self.args.momentum,  weight_decay=self.args.weight_decay)
        lr_scheduler = LearningRateScheduler(self.args.lr_wr_epochs, len(dataset.train_loader.dataset), self.args.batch_size,
                                             self.args.learning_rate, self.args.lr_wr_mul, self.args.lr_wr_min)

        train_flag = True
        epoch = 0
        while epoch < self.args.epochs:
            train(dataset, model, criterion, epoch, optimizer, lr_scheduler, self.device, self.args)
            prec = validate(dataset, model, criterion, epoch, self.device, self.args)
            best_prec = max(prec, best_prec)
            # TODO: hard-coded early stopping criterion of last prec < 15%
            if epoch==(self.args.lr_wr_epochs - 1) and float(prec)<(1.5 *100./10):
                train_flag = False
                break
            epoch += 1
        if self.args.no_gpus>1:
            spp_filter_size = model.module.spp_filter_size
        else:
            spp_filter_size = model.spp_filter_size
        del model, criterion, optimizer, lr_scheduler
        return spp_filter_size, best_prec, train_flag

    def __reset_for_new_walk(self):

        self.state_list = []
        self.bucketed_state_list = []
        self.state = se.State('start', 0, 1, 0, 0, self.args.patch_size, 0, 0)

    def __run_agent(self):

        while self.state.terminate == 0:
            self.__transition_q_learning()

        return self.state_list

    def __transition_q_learning(self):
        if self.state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.state, self.qstore.q)        
        action_values = self.qstore.q[self.state.as_tuple()]
        if np.random.random() < self.epsilon:
            action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i] == max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        self.__post_transition_updates()

    def __post_transition_updates(self):
        non_bucketed_state = self.state.copy()
        self.state_list.append(non_bucketed_state)

    def sample_replay_for_update(self):
        net = self.replay_dictionary.iloc[-1]['net']
        reward_best_val = self.replay_dictionary.iloc[-1]['reward']
        train_flag = self.replay_dictionary.iloc[-1]['train_flag']
        state_list = self.stringutils.convert_model_string_to_states(cnn_parse('net', net))
        # if train_flag:
        self.__update_q_value_sequence(state_list, self.__accuracy_to_reward(reward_best_val/100.))

        for i in range(self.state_space_parameters.replay_number-1):
            net = np.random.choice(self.replay_dictionary['net'])
            reward_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['reward'].values[0]
            train_flag = self.replay_dictionary[self.replay_dictionary['net'] == net]['train_flag'].values[0]
            state_list = self.stringutils.convert_model_string_to_states(cnn_parse('net', net))
            # if train_flag == True:
            self.__update_q_value_sequence(state_list, self.__accuracy_to_reward(reward_best_val/100.))            

    def __accuracy_to_reward(self, acc):
        return acc

    def __update_q_value_sequence(self, states, termination_reward):
        self.__update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            
            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            self.__update_q_value(states[i], states[i+1], 0)

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
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

        # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                       self.args.q_learning_rate * \
                                                       (reward + self.args.q_discount_factor *
                                                        max_over_next_states -
                                                        values[actions.index(action_between_states)])

        # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
        # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
        #                                                (max(reward, values[actions.index(action_between_states)]) -
        #                                                 values[actions.index(action_between_states)])

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
