"""
Counterfactual learning

# debugging:
ipython cf_learning/main.py

"""

from PlanB_model import CoPhyNet
from torch import optim
from torch.utils.data import DataLoader
import torch
import numpy as np
# import argparse
import ipdb
import os
from tqdm import *
from random import choice
import torch.nn.functional as F
import time
# from dataloaders.utils import *
from causal_world.task_generators import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy

def Calc_num_objects(env_obs):
    len_obs = env_obs.shape[0]
    # print("length of obs: ", len_obs)
    num_of_objects = int(len_obs/28) - 1
    # print("num of objects: ", num_of_objects)
    return num_of_objects

def Convert_input_shape(stack_obs, timesteps, num_of_objects):
    # 28 len vector for T,R1,R2,R3 in struct obs of CausalWorld and ...
    # Another 28 for each object (object features, partial goal features) ...
    # for struct obs
    desired_input_obs = np.empty([timesteps, num_of_objects, 28 + 28])
    lens_obs = stack_obs.shape[1]

    for t in range(timesteps):
        for k in range(num_of_objects):
            # get T, R1, R2, R3 which is same for all objects
            desired_input_obs[t, k, :28] = stack_obs[t, :28]
            # Get features for each object
            obj_index_lower = 28 + (k * 28)
            obj_index_upper = obj_index_lower + 28
            if obj_index_upper >= lens_obs:
                # print("convert input shape obj_index_lower: ", obj_index_lower)
                desired_input_obs[t, k, 28:] = stack_obs[t, obj_index_lower:]
            else:
                desired_input_obs[t, k, 28:] = stack_obs[t, obj_index_lower:obj_index_upper]

    return desired_input_obs


def main():
    # init gpu for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_loader = {'batch_size': 32}
    if device.type == 'cuda':
        kwargs_loader.update({'num_workers': 10, 'pin_memory': True})

    # init RL environment and model
    task = generate_task(task_generator_id="pushing", 
                            tool_block_mass=0.3)
    env = CausalWorld(task=task, skip_frame = 1, enable_visualization=False)
    RL_model = PPO2(MlpPolicy, env, verbose=1)

    # Init and calc some values
    obs = env.reset()
    num_of_objects = Calc_num_objects(obs)
    stack_input_obs = np.empty(28 + (num_of_objects * 28))
    num_timesteps = 3

    # Obtain sequence of observations to train CF model
    for i in range(num_timesteps):
        obs = env.reset()
        if i == 0:
            stack_input_obs = np.expand_dims(obs, axis=0)
        elif i > 0:
            stack_input_obs = np.concatenate((stack_input_obs, np.expand_dims(obs, axis=0)), axis=0)
    # print("stack input obs: ", stack_input_obs)

    # Get observations ab for CF model
    desired_input_obs = Convert_input_shape(stack_input_obs, num_timesteps, num_of_objects)
    tensor_input_obs_ab = torch.from_numpy(desired_input_obs)
    # print("tensor input obs ab: ", tensor_input_obs_ab.shape)

    # Get observations c for CF model
    goal_intervention_dict = env.sample_new_goal()
    success_signal, intervene_obs_c = env.do_intervention(goal_intervention_dict)
    print("Goal Intervention success signal", success_signal)
    intervene_obs_c = np.expand_dims(intervene_obs_c, axis=0)
    desired_input_obs = Convert_input_shape(intervene_obs_c, 1, num_of_objects)
    tensor_input_obs_c = torch.from_numpy(desired_input_obs)
    # print("tensor input obs c: ", tensor_input_obs_c.shape)

    # Send to CF model
    CF_model = CoPhyNet(num_objects=num_of_objects)
    CF_model_out, CF_model_stab = CF_model.forward(tensor_input_obs_ab, tensor_input_obs_c)

if __name__ == "__main__":
    main()
