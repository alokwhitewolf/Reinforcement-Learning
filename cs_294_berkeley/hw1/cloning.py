# To clone the behaviour of an expert using Neural Network
'''

'''

import numpy as np
import gym
import tensorflow as tf
import tf_util
import pickle
import load_policy

import time

import argparse

from keras.models import load_model,Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.utils import np_utils
from keras import losses

from sklearn.utils import shuffle








def get_rollout_data(filename):
	
	with open(filename, 'rb') as f:
		rollout_data = pickle.loads(f.read())
	return rollout_data


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('data_file', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20,
	                    help='Number of expert roll outs')
	args = parser.parse_args()

	expert_policy = load_policy.load_policy(args.expert_policy_file)
	num_rollouts = args.num_rollouts
	env_name = args.envname
	env_data = get_rollout_data(args.data_file)

	obs_data = np.asarray(env_data['observations']) ##Input for Neural Net
	action_data = np.asarray(env_data['actions']) ##Output for Neural Net

	obs_data, action_data = shuffle(obs_data, action_data)


	n = obs_data.shape[0]

	split_on = int(n*.8)

	X_train = np.array(obs_data[:split_on])
	X_test = np.array(obs_data[split_on:])
	y_train = np.array(action_data[:split_on])
	y_test = np.array(action_data[split_on:])

	X_train = X_train.reshape(X_train.shape[0], obs_data.shape[1])
	X_test = X_test.reshape(X_test.shape[0], obs_data.shape[1])
	Y_train = y_train.reshape(y_train.shape[0], action_data.shape[2])
	Y_test = y_test.reshape(y_test.shape[0], action_data.shape[2])
	print (obs_data.shape)
	print (action_data.shape)
	
	## Start training
	# Create a feedforward neural network


	
	
	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=(obs_data.shape[1],)))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(action_data.shape[2], activation='linear'))
	

	#Explain why msle
	model.compile(loss=losses.mean_squared_logarithmic_error, optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1)
	score = model.evaluate(X_test, Y_test, verbose=1)

	model.save('cloned_models/' + env_name +'_'+str(args.num_rollouts)+ '_cloned_model.h5')
	

	with tf.Session():
		tf_util.initialize()
		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit

		returns = []
		#load model
		model = load_model('cloned_models/' + env_name +'_'+str(args.num_rollouts)+ '_cloned_model.h5')
		for i in range(args.num_rollouts):
			#print('iter', i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				obs = np.array(obs)
				
				
				obs = obs.reshape(1, len(obs))
				#predict action using loaded model
				action = (model.predict(obs, batch_size=64, verbose=0))
				# print 'obs: ' + str(obs)
				# print "predicted: " + str(action)
				# print "expert: " + str(exp_action)

				obs, r, done, _ = env.step(action)
				totalr += r
				
				steps += 1
				if args.render:
					env.render()
					time.sleep(0.01)
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			returns.append(totalr)
		with open("cloning_results.txt","a") as file:
			file.write("\n ")
			file.write("\n num of rollouts "+str(num_rollouts))
			file.write("\n Total returns : "+str(sum(returns)))
			file.write("\n mean return : "+str(np.mean(returns)))
			file.write("\n std of return ; "+str(np.std(returns)))

if __name__ == '__main__':
	main()