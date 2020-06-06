import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(path):
    return np.load(path)

def moving_avg(arr, n):
	moving_average = []
	for i in range(len(arr) - n + 1):
		moving_average.append(np.mean([arr[i:i+n]]))
	return moving_average

def plot_reward(reward_list, n):


	for i in reward_list:
		data = read_data(i)
		mv_avg = moving_avg(data, n)
		plt.plot(range(len(mv_avg)), mv_avg)

def get_gamma(path):
	return path.split('gamma-')[1].split('.npy')[0]

def main():
	print('Loading..')

	# plot Q1
	plt.figure(1)
	plt.suptitle('pg learning curve (n=20)')
	plt.xlabel('episode')
	plt.ylabel('reward')
	plot_reward(['reward/pg-reward.npy'], 20)
	plt.show()

	plt.figure(1)
	plt.suptitle('dqn learning curve (n=300)')
	plt.xlabel('episode')
	plt.ylabel('reward')
	plot_reward(['reward/dqn-reward.npy'], 300)
	plt.show()

	# plot Q2
	path = ['reward/dqn-reward-gamma-0.1.npy', 'reward/dqn-reward-gamma-0.95.npy', 'reward/dqn-reward-gamma-0.99.npy', 'reward/dqn-reward-gamma-1.02.npy']
	plt.figure(1)
	plt.suptitle('dqn-learning curve (hyperparam=GAMMA)')
	plt.xlabel('episode')
	plt.ylabel('reward')
	plot_reward(path, 300)
	for i in range(len(path)):
		path[i] = get_gamma(path[i])
	plt.legend(path)
	plt.show()

	# plot Q3
	path = ['reward/pg-reward.npy', 'reward/pg-vr-reward.npy']
	plt.figure(1)
	plt.suptitle('pg improvement')
	plt.xlabel('episode')
	plt.ylabel('reward')
	plot_reward(path, 30)
	plt.legend(['pg', 'pg w/vr'])
	plt.show()

	path = ['reward/dqn-reward.npy', 'reward/ddqn-reward.npy']
	plt.figure(1)
	plt.suptitle('dqn improvement')
	plt.xlabel('episode')
	plt.ylabel('reward')
	plot_reward(path, 500)
	plt.legend(['dqn', 'ddqn'])
	plt.show()





if __name__ == '__main__':
	main()
