import os
import numpy as np
from causal_world.task_generators import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK, REACHING_BENCHMARK, PICKING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis
from stable_baselines import PPO2, SAC

def main():
    block_size_iter = 4
    block_mass_iter = 4
    block_color_iter = 4
    floor_friction_iter = 4
    floor_color_iter = 4
    robot_finger_mass_iter = 4

    # Path for own Causal RL model
    CFRL_model_path = "./trained_CFRL_model_push_pick/model_CFRL_7000000_steps.zip"
    CausalRep_path = "./trained_CFRL_model_iter/CausalRep_2.npy"

    # Evaluation pipeline
    task_params = dict()
    task_params['task_generator_id'] = 'picking'
    world_params = dict()
    world_params['skip_frame'] = 3
    evaluation_protocols = PICKING_BENCHMARK['evaluation_protocols']
    evaluator_1 = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                    task_params=task_params,
                                    world_params=world_params,
                                    visualize_evaluation=False)

    # Load model from model_path
    CausalRL_model = SAC.load(CFRL_model_path)
    CausalRep = np.load(CausalRep_path)

    def CFRL_policy_fn(obs):
        CausalRL_obs = np.append(obs, CausalRep)
        return CausalRL_model.predict(CausalRL_obs, deterministic=True)[0]
    
    scores_model_CFRL = evaluator_1.evaluate_policy(CFRL_policy_fn, fraction=1.0)
    experiments = dict()
    experiments['Causal_SAC'] = scores_model_CFRL
    vis.generate_visual_analysis('./log_dir/', experiments=experiments)

if __name__ == "__main__":
    main()