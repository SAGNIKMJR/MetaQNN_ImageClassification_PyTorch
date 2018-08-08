import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='plotting rolling mean from replay dict from MetaQNN')
parser.add_argument('--csv', metavar='CSV', default='./runs/replayDictFinal.csv' , 
                    help='path to csv sorted according to epsilon in descending\
                     order to plot rolling mean from')
parser.add_argument('-rc', '--reward-col', metavar='RC', default='reward',\
					 help='column name to take rewards from')
parser.add_argument('-rmw', '--rolling-mean-window', metavar='RMW', type=int, default=None,\
					 help='window size to calculate the rolling mean')

def plotRollingMean(args):
	df_replayDict = pd.read_csv(args.csv)
	epsilon_replayDict = df_replayDict['epsilon'].tolist()
	reward_replayDict = df_replayDict[args.reward_col].tolist()
	rollingMeanReward_replayDict = list()
	if args.rolling_mean_window is None:
		for i in range(0,len(reward_replayDict)):
			rollingMeanReward_replayDict.append(np.mean(np.array(reward_replayDict[0:i+1])))
	else:
		for i in range(int(args.rolling_mean_window),len(reward_replayDict)):
			rollingMeanReward_replayDict.append(np.mean(\
										np.array(reward_replayDict[i-int(args.rolling_mean_window):i+1])))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(rollingMeanReward_replayDict)

	# TODO: add ticks and markers
	"""
		ax.plot(reward_replayDict)
		x_ticks = np.append(ax.get_xticks(),[33,41,50,59,66,79,91,102,112,118])
		x = np.arange(1,len(reward_replayDict)+1) 
		plt.xticks(x,epsilon_replayDict)
		ax.set_xticks(x_ticks)
		plt.plot(x,rollingMeanReward_replayDict)
	"""
	plt.ylabel(args.reward_col)
	plt.show()
if __name__ == "__main__":
	args = parser.parse_args()
	plotRollingMean(args) 