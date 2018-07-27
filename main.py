import torch
import os
import sys
import numpy as np
from time import gmtime, strftime
import pandas as pd

from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
from lib.MetaQNN.q_learner import QLearner as QLearner
import lib.Models.model_state_space_parameters as state_space_parameters
from lib.Models.initialization import WeightInit
# TODO: gives interrupted system call error
# from lib.Utility.auxiliary_utils import Logger


def main():
	# Check whether GPU is available and can be used
	# if CUDA is found then device is set accordingly
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	save_path = './runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	log_file = os.path.join(save_path, "stdout")
	log = open(log_file, "a")
	# TODO: gives interrupted sys call error
	# log_file = os.path.join(save_path, "stdout")
	# sys.stdout = Logger(log_file)

	# Command line options
	args = parser.parse_args()
	print("Command line options:")
	for arg in vars(args):
		print(arg, getattr(args, arg))
		log.write(arg + ':' + str(getattr(args, arg)) + '\n')
	log.close()

	# Initialize the weights of the model
	print("Initializing network with: " + args.weight_init)
	WeightInitializer = WeightInit(args.weight_init)

	# Dataset loading
	# TODO: hard-coded file paths
	patch_size = args.patch_size                
	data_init_method = getattr(datasets, args.dataset)
	dataset = data_init_method(torch.cuda.is_available(), args)

	gen = QLearner(state_space_parameters, 1, WeightInitializer, device, args, save_path, qstore = args.qstore_path, replaydict = args.replay_dict_path)

	if (args.continue_epsilon not in np.array(state_space_parameters.epsilon_schedule)[:,0]):
		raise ValueError('continue-epsilon {} not in epsilon schedule!'.format(args.continue_epsilon))

	for episode in state_space_parameters.epsilon_schedule:

		epsilon = episode[0]
		M = episode[1]

		for ite in range(1,M+1):
			if epsilon == args.continue_epsilon and args.continue_ite > M:
				raise ValueError('continue-ite {} not within range of continue-epsilon {} in epsilon schedule!'.format(args.continue_ite, epsilon))
			if (epsilon == args.continue_epsilon and ite >= args.continue_ite) or (epsilon<args.continue_epsilon):
				print('ite:{}, epsilon:{}'.format(ite, epsilon))
				gen.generate_net(epsilon, dataset)


	gen.replay_dictionary.to_csv(os.path.join(save_path, 'replayDictFinal.csv'))
	gen.qstore.save_to_csv(os.path.join(save_path, 'qValFinal.csv'))


if __name__ == '__main__':
	main()
