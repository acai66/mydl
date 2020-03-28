# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
import os.path
import pickle
from time import time
from collections import deque 

# Hyper Parameters:
FRAME_PER_ACTION = 2
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 200. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.33 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 100 # size of minibatch
UPDATE_TIME = 100
PRINT_ITER = 50
learning_rate = 5e-6
model_save_path = os.path.join('saved_networks', 'network-dqn.pkl')
modelT_save_path = os.path.join('saved_networks', 'network-dqnT.pkl')


class Model(tf.Module):

	def __init__(self, actions):
		# print(actions.dtype)
		self.cost = 0
		self.actions = actions

		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.optimizer = tf.optimizers.SGD(learning_rate)
		self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()


	def createQNetwork(self):
		# network weights
		W_conv1 = self.weight_variable([8,8,4,32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4,4,32,64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3,3,64,64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([1600,512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		return W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	def QNetwork(self, stateInput):

		
		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(stateInput,self.W_conv1,4) + self.b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1,self.W_conv2,2) + self.b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2,self.W_conv3,1) + self.b_conv3)

		h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,self.W_fc1) + self.b_fc1)

		# Q Value layer
		QValue = tf.matmul(h_fc1,self.W_fc2) + self.b_fc2
		
		return QValue


	def QNetworkCost(self, stateInput, action_input, y_Input):
		self.QValue = self.QNetwork(stateInput)
		self.actionInput = action_input
		self.yInput = y_Input
		Q_Action = tf.reduce_sum(tf.math.multiply(self.QValue, self.actionInput), axis = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))

		return self.cost



	def trainStep(self, stateInput, action_input, y_Input):
		with tf.GradientTape() as g:
			self.cost = self.QNetworkCost(stateInput, action_input, y_Input)
		gradients = g.gradient(self.cost,[self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2])
		self.optimizer.apply_gradients(zip(gradients,[self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2]))



	def trainQNetwork(self, modelT, replayMemory):

		
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []

		QValue_batch = modelT.QNetwork(nextState_batch)
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.trainStep(state_batch, action_batch, y_batch)



	def weight_variable(self,shape):
		initial = tf.random.normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)


	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		



class BrainDQN:

	def __init__(self,actions):
		
		
		# init replay memory
		self.replayMemory = deque()
		# init Q network and Target Q Network
		# load saved models
		if os.path.exists(model_save_path) and os.path.exists(modelT_save_path):
			self.model = pickle.load(open(model_save_path, 'rb'))
			self.modelT = pickle.load(open(modelT_save_path, 'rb'))
		else:
			self.model = Model(actions)
			self.modelT = Model(actions)



	def copyTargetQNetwork(self):
		self.modelT.W_conv1.assign(self.model.W_conv1)
		self.modelT.b_conv1.assign(self.model.b_conv1)
		self.modelT.W_conv2.assign(self.model.W_conv2)
		self.modelT.b_conv2.assign(self.model.b_conv2)
		self.modelT.W_conv3.assign(self.model.W_conv3)
		self.modelT.b_conv3.assign(self.model.b_conv3)
		self.modelT.W_fc1.assign(self.model.W_fc1)
		self.modelT.b_fc1.assign(self.model.b_fc1)
		self.modelT.W_fc2.assign(self.model.W_fc2)
		self.modelT.b_fc2.assign(self.model.b_fc2)



		
	def setPerception(self,nextObservation,action,reward,terminal):
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = tf.concat([self.currentState[:,:,1:],nextObservation],axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.model.timeStep > OBSERVE and len(self.replayMemory) >= BATCH_SIZE:
			# Train the network
			start_train_time = time()
			self.model.trainQNetwork(self.modelT, self.replayMemory)
			end_train_time = time()
			train_time = end_train_time - start_train_time
			# save network every 5000 iteration
			if self.model.timeStep % 5000 == 0:
				pickle.dump(self.model, open(model_save_path, 'wb'))
				pickle.dump(self.modelT, open(modelT_save_path, 'wb'))

			if self.model.timeStep % UPDATE_TIME == 0:
				self.copyTargetQNetwork()

		# print info
		state = ""
		if self.model.timeStep <= OBSERVE or len(self.replayMemory) <= BATCH_SIZE:
			state = "observe"
		elif self.model.timeStep > OBSERVE and self.model.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"
		if self.model.timeStep % PRINT_ITER == 0:
			print ("TIMESTEP", self.model.timeStep, "/ STATE", state, "/ LOSS", self.model.cost, "/ EPSILON", self.model.epsilon)
		
		self.currentState = newState
		self.model.timeStep += 1
		return self.model.timeStep, self.model.cost


	def getAction(self):
		stateInput = tf.reshape(self.currentState, [-1, 80, 80, 4])
		QValue = self.model.QNetwork(stateInput)[0]
		action = np.zeros(self.model.actions)
		action_index = 0
		if self.model.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.model.epsilon:
				action_index = random.randrange(self.model.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1 # do nothing

		# change episilon
		if self.model.epsilon > FINAL_EPSILON and self.model.timeStep > OBSERVE:
			self.model.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		return action


	def setInitState(self,observation):
		self.currentState = tf.stack((observation, observation, observation, observation), axis = 2)


		
