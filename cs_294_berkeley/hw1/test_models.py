# To clone the behaviour of an expert using Neural Network

import numpy as np
import gym
import tensorflow as tf
import tf_util
import load_policy

import time
import argparse
from keras.models import load_model
import keras.backend as K

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('model_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	
	args = parser.parse_args()

	env_name = args.envname
	model_file = ('Dagger_models/Humanoid-v1'+'_'+ '25' +'_2_'+'_Dagger_model.h5')



	
	model = load_model(model_file)

	
	with tf.Session():
		tf_util.initialize()
		K.get_session().run(tf.initialize_all_variables())

		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit


	
		for i in range(15):
			#print('iter', i)
			
			obs = env.reset()
			done = False
			returns = 0.
			steps = 0
			while not done:
				obs = np.array(obs)
				print ("AAAAAAAAAA")
				
				obs = obs.reshape(1, len(obs))
				action = (model.predict(obs, batch_size=64, verbose=0))
				

				
				obs, r, done, _ = env.step(action)
				returns += r
				steps += 1
				if args.render:
					env.render()
					time.sleep(0.1)
					
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
					
			print('returns', returns)
			print('mean return', np.mean(returns))
			print('std of return', np.std(returns))

if __name__ == '__main__':
	main()