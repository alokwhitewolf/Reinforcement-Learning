# Deep Reinforcement Learning Algorithms
This section contains implementations of various deep reinforcement learning algorithms taught as part CS 294-112, [UC Berkeley's Deep Reinforcement Learning course](http://rll.berkeley.edu/deeprlcoursesp17/).

## Dependencies
The dependencies of the algorithms include:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [MuJoCo](http://www.mujoco.org/) [Paid library, but there is a free student license]
- [Roboschool](https://github.com/openai/roboschool) [Free alternative to MuJoCo being set up by OpenAI]

## HW1: Imitation Learning and DAgger
Implemented [behavior cloning](http://rll.berkeley.edu/deeprlcourse/docs/week_2_lecture_1_behavior_cloning.pdf) on multiple MuJoCo environments. Expert policies produce rollouts that are used as training data for a feedforward neural network. 

Also implemented the [DAgger](http://rll.berkeley.edu/deeprlcourse-fa15/docs/2015.10.5.dagger.pdf) algorithm, which performs significantly better. 