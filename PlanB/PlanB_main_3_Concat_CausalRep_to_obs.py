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
from causal_world.actors.pushing_policy import PushingActorPolicy
from causal_world.actors.stacking2_policy import Stacking2ActorPolicy
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
            # Get features for each object 17 dim
            obj_index_lower = 28 + (k * 17)
            obj_index_upper = obj_index_lower + 17
            desired_input_obs[t, k, 28:45] = stack_obs[t, obj_index_lower:obj_index_upper]
            # Get partial goal feature of each object 11 dim, concat after the object features
            part_goal_ind_low = (28 + (num_of_objects*17)) + (k*11)
            part_goal_ind_up = part_goal_ind_low + 11
            if part_goal_ind_up >= lens_obs:
                # print("convert input shape part_goal_ind_low: ", part_goal_ind_low)
                desired_input_obs[t, k, 45:] = stack_obs[t, part_goal_ind_low:]
            else:
                desired_input_obs[t, k, 45:] = stack_obs[t, part_goal_ind_low:part_goal_ind_up]


    return desired_input_obs

def Calc_loss(pred_obs_d, actual_obs_d):
    total_num_features = pred_obs_d.shape[1] * pred_obs_d.shape[2] * pred_obs_d.shape[3]
    mse_3d = torch.sum(((pred_obs_d - actual_obs_d) ** 2).mean(-1)) / total_num_features  # (B,K)
    return mse_3d

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
    pretrained_RL_agent = PushingActorPolicy()

    # Init and calc some values
    obs = env.reset()
    num_of_objects = Calc_num_objects(obs)
    CF_model = CoPhyNet(num_objects=num_of_objects).to(device)

    # Pre-training of CF model with pretrained RL agent
    optimizer = optim.Adam(CF_model.parameters(), lr=1e-3)
    print_freq = 5
    log_file_train = "./log_dir/log_train.txt"
    list_mse_3d = []
    num_timesteps = 30
    max_iter = 40

    CF_model.train()
    for curr_iter in range(max_iter):
        start_iter_time = time.time()
        stack_input_obs = np.empty(28 + (num_of_objects * 28))

        # Obtain sequence of observations to train CF model
        action = pretrained_RL_agent.act(obs)
        # print("agent action: ", action)
        for i in range(num_timesteps):
            next_obs, _, _, _ = env.step(action)
            action = pretrained_RL_agent.act(next_obs)
            if i == 0:
                stack_input_obs = np.expand_dims(next_obs, axis=0)
            elif i > 0:
                stack_input_obs = np.concatenate((stack_input_obs, np.expand_dims(next_obs, axis=0)), axis=0)
        # print("stack input obs: ", stack_input_obs)

        # Get observations ab for CF model
        desired_input_obs = Convert_input_shape(stack_input_obs, num_timesteps, num_of_objects)
        tensor_input_obs_ab = torch.from_numpy(desired_input_obs).to(device)
        # print("tensor input obs ab: ", tensor_input_obs_ab.shape)

        # Get observations c for CF model
        goal_intervention_dict = env.sample_new_goal()
        success_signal, intervene_obs_c = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention for obs c success signal", success_signal)
        action = pretrained_RL_agent.act(intervene_obs_c)
        intervene_obs_c = np.expand_dims(intervene_obs_c, axis=0)
        desired_input_obs = Convert_input_shape(intervene_obs_c, 1, num_of_objects)
        tensor_input_obs_c = torch.from_numpy(desired_input_obs).to(device)
        # print("tensor input obs c: ", tensor_input_obs_c.shape)

        # Send to CF model
        CF_model_out, CF_model_stab, CF_causal_rep = CF_model.forward(tensor_input_obs_ab, tensor_input_obs_c)

        # Testing the Calc_loss function
        stack_obs_cd = np.empty(28 + (num_of_objects * 28))
        # print("env sample action: ", env.action_space.sample()) # To check true shape of action
        # print("agent action: ", action)
        for i in range(num_timesteps-1):
            next_obs, _, _, _ = env.step(action)
            action = pretrained_RL_agent.act(next_obs)
            # print("agent action: ", action)
            if i == 0:
                stack_obs_cd = np.expand_dims(next_obs, axis=0)
            elif i > 0:
                stack_obs_cd = np.concatenate((stack_obs_cd, np.expand_dims(next_obs, axis=0)), axis=0)
        # print("stack obs cd shape: ", stack_obs_cd.shape)
        desired_input_obs = Convert_input_shape(stack_obs_cd, num_timesteps-1, num_of_objects)
        tensor_actual_obs_d = torch.from_numpy(desired_input_obs)
        # print("tensor actual obs d: ", tensor_actual_obs_d.shape)
        actual_obs_d = tensor_actual_obs_d.unsqueeze(0).to(device)
        # print("actual obs d: ", actual_obs_d.shape)

        # loss
        mse_3d = Calc_loss(CF_model_out, actual_obs_d)

        # backprop
        optimizer.zero_grad()
        mse_3d.backward()
        optimizer.step()

        goal_intervention_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)

        end_iter_time = time.time()
        iter_time = end_iter_time - start_iter_time

        print("Current iter: ", curr_iter)
        print("Max iter: ", max_iter)
        print(f"time for this iter: {iter_time:.3f}")
        if curr_iter % print_freq == 0:
            print(f"curr iter mse loss: {mse_3d:.6f}")
            list_mse_3d.append(mse_3d.item())
            print(f"Mean of mse_3d over {print_freq} iters: {np.mean(list_mse_3d):.6f}")

    with open(log_file_train, "+a") as f:
        f.write(f"Mean of mse_3d over 1 epoch: {np.mean(list_mse_3d):.6f}")

    # TRY CONCAT CAUSAL REP TO OBS FOR TRAIN RL AGENT
    numpy_causal_rep = CF_causal_rep.detach().cpu().numpy()
    numpy_causal_rep = np.squeeze(numpy_causal_rep, axis=0)
    #print("numpy_causal_rep shape: ", numpy_causal_rep.shape)
    # For passing to environment
    min_val_causal_rep = np.amin(numpy_causal_rep)
    max_val_causal_rep = np.amax(numpy_causal_rep)
    shape_causal_rep = numpy_causal_rep.shape
    #print(f"low of CRep= {min_val_causal_rep} and high of CRep= {max_val_causal_rep}")

    # Delete the env that has been created first
    del(env)


    # init new RL environment and model
    task = generate_task(task_generator_id="pushing", 
                            tool_block_mass=0.3, mode=1, causal_rep=numpy_causal_rep,
                            low_cRep=min_val_causal_rep, high_cRep=max_val_causal_rep,
                            shape_cRep=shape_causal_rep)
    env = CausalWorld(task=task, skip_frame = 1, enable_visualization=False)
    RL_model = PPO2(MlpPolicy, env, verbose=1)
    RL_model.learn(total_timesteps=25000)

        

if __name__ == "__main__":
    main()
