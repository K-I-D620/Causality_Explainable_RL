import tensorflow as tf
import os
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

def main():
    # init RL environment and model
    task = generate_task(task_generator_id="pushing", 
                            tool_block_mass=0.3)
    env = CausalWorld(task=task, skip_frame = 1, enable_visualization=False)

    log_relative_path = "./log_dir"
    total_time_steps=50000
    validate_every_timesteps=25000
    # Parameters for RL model
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    ppo_config = {
        "gamma": 0.9995,
        "n_steps": 5000,
        "ent_coef": 0,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 10,
        "nminibatches": 1000,
        "noptepochs": 4
    }

    # Create RL model with parameters and train
    RL_model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, **ppo_config)
    for _ in range(int(total_time_steps / validate_every_timesteps)):
        RL_model.learn(total_timesteps=validate_every_timesteps,
                        reset_num_timesteps=False)
        RL_model.save(os.path.join(log_relative_path, 'saved_model'))

if __name__ == "__main__":
    main()
