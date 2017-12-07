'''
Code for implementing DAgger algorithm

example usage -- python3 Dagger.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
'''


import numpy as np
import gym
import tensorflow as tf
import tf_util
import pickle
import load_policy

import time

import argparse

import keras.backend as K
from keras.models import load_model,Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.utils import np_utils
from keras.callbacks import History
from keras import losses
from keras import optimizers

from sklearn.utils import shuffle
from matplotlib import pyplot as plt
plt.style.use(['ggplot'])








def get_rollout_data(filename):
	
	with open(filename, 'rb') as f:
		rollout_data = pickle.loads(f.read())
	return rollout_data


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('data_file', type=str)#To train the model initially
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20,
	                    help='Number of expert roll outs')
	args = parser.parse_args()

	expert_policy = load_policy.load_policy(args.expert_policy_file)

	env_name = args.envname
	env_data = get_rollout_data(args.data_file)

	obs_data = np.asarray(env_data['observations']) ##Input for Neural Net 
	action_data = np.asarray(env_data['actions']) ##Output for Neural Net

	num_rollouts = args.num_rollouts

	##Live plotting of data
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	fig.suptitle(str(env_name), fontsize=16)
	
	plt.title(str(env_name))
	
	plt.ion()
	


	
	##Create the feedforward model

	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=(obs_data.shape[1],)))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(action_data.shape[2], activation='linear'))
	model.compile(loss=losses.mean_squared_logarithmic_error, optimizer="adam", metrics=['accuracy'])
	
	
	#Avg returns of each model
	total_returns = []

	#Keeps counter of number
	counter = 0

	#Keeps record for each model iteration (for plotting)
	counters = []
	rewards_to_plot = []
	avg_rewards_to_plot = []

	model_rewards_list = []
	model_counters_list = []

	#Store current mode(that is being tested) values,
	#Also to calculate averages for each model
	current_model_counter = []
	current_model_rewards = []

	#To save model only if the eval metrics improve
	best_eval = 0.
	Wsave = 0.
	first = True

	#Main DAgger loop
	for i in range(25):

		with tf.Session():
			
			n = obs_data.shape[0]

			#Split into test and and valid set
			split_on = int(n*.80)

			obs_data, action_data = shuffle(obs_data, action_data)

			X_train = np.array(obs_data[:])
			X_test = np.array(obs_data[split_on:])
			y_train = np.array(action_data[:])
			y_test = np.array(action_data[split_on:])

			X_train = X_train.reshape(X_train.shape[0], obs_data.shape[1])
			X_test = X_test.reshape(X_test.shape[0], obs_data.shape[1])
			Y_train = y_train.reshape(y_train.shape[0], action_data.shape[2])
			Y_test = y_test.reshape(y_test.shape[0], action_data.shape[2])

			
			
			if first:
				#First time training. 
				K.get_session().run(tf.initialize_all_variables())
				#Train
				model.fit(X_train, Y_train, batch_size=64, nb_epoch=40, verbose=1)
				
				first = False
			else:
				
				#Intialize weights from prev best model
				K.get_session().run(tf.initialize_all_variables())
				model.set_weights(Wsave)
				#Train
				model.fit(X_train, Y_train, batch_size=64, nb_epoch=40, verbose=1)
				

			score = model.evaluate(X_test, Y_test, verbose=1)
			best_eval = max(best_eval,score[1])

			#Initiate env
			env = gym.make(args.envname)
			max_steps = args.max_timesteps or env.spec.timestep_limit

			#Reset lists
			returns = []
			new_observations = []
			new_exp_actions = []

			current_model_counter = []
			current_model_rewards = []

			
			for i in range(args.num_rollouts):
				#print('iter', i)
				obs = env.reset()
				done = False
				#totalr --> return for particular episode
				totalr = 0.
				steps = 0
				while not done:
					obs = np.array(obs)

					exp_action = expert_policy(obs[None,:])
					obs = obs.reshape(1, len(obs))
					action = (model.predict(obs, batch_size=64, verbose=0))
					
					#Add the obs and action data for DAgger
					if exp_action is not action:
						new_observations.append(obs)
						new_exp_actions.append(exp_action)
					obs, r, done, _ = env.step(action)
					totalr += r
					steps += 1
					if args.render:
						env.render()
						
						
						
					#if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
					if steps >= max_steps:
						break
				#Append to respective lists
				returns.append(totalr)
				counter +=1
				counters.append(counter)
				current_model_counter.append(counter)

				rewards_to_plot.append(totalr)
				current_model_rewards.append(totalr)
				avg_rewards_to_plot.append(np.mean(rewards_to_plot))
				
				#plot
				ax1.clear()
				ax1.plot(counters, rewards_to_plot,'b')
				ax1.plot(counters,avg_rewards_to_plot,'g')
				for c, r in zip(model_counters_list,model_rewards_list):
					ax1.plot(c, [np.mean(r)]*len(c),'r')

				plt.pause(0.001)
				plt.show()
				fig.savefig(str(env_name)+'.png') 

			
			total_returns.append(np.mean(returns))
			#print('returns', returns)
			#print('mean return', np.mean(returns))
			#print('std of return', np.std(returns))

			#mean_rewards.append(np.mean(returns))
			#stds.append(np.std(returns))
			model_counters_list.append(current_model_counter)
			model_rewards_list.append(current_model_rewards)
			

			ax1.clear()
			ax1.plot(counters, rewards_to_plot,'b')
			ax1.plot(counters,avg_rewards_to_plot,'g')
			for c, r in zip(model_counters_list,model_rewards_list):
				ax1.plot(c, [np.mean(r)]*len(c),'r')

			plt.pause(0.001)
			plt.show()
			fig.savefig(str(env_name)+'.png') 

			#If the current model gives best result, save weight
			if np.mean(model_rewards_list[-1]) >= best_eval:
				Wsave = model.get_weights()

				model.save('Dagger_models/' + env_name +'_'+ str(num_rollouts) +'_Dagger_model.h5',overwrite=True)
				best_eval = np.mean(model_rewards_list[-1])
				print ("####SAVED####")

			new_observations = np.array(new_observations)
			new_exp_actions = np.array(new_exp_actions)
		
		#Add data for  DAgger
		new_observations = new_observations.reshape((new_observations.shape[0], obs_data.shape[1]))

		obs_data = np.concatenate((obs_data, new_observations))
		action_data = np.concatenate((action_data, new_exp_actions))
	
	with open("DAgger_results.txt","a") as file:
		file.write("\n ")
		file.write("\n num of rollouts "+str(num_rollouts))
		file.write("\n Total returns : "+str(sum(total_returns)))
		file.write("\n mean return : "+str(np.mean(total_returns)))
		file.write("\n std of return ; "+str(np.std(total_returns)))


if __name__ == '__main__':
	main()