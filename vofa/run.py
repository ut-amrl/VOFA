import argparse
import yaml

import isaacgym
from runner import (PPO, DAgger, FalconPPO, FalconCamEvaluate, FalconDAgger)

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True, type=str, help="Path to the runner class to run.", choices=["FalconPPO", "FalconCamEvaluate", "FalconDAgger", "FalconDAggerFromStudent", "PPO", "DAgger"])
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--eval", action="store_true", help="Run in eval mode.")
    parser.add_argument("--config", required=True, type=str, help="Path to the config file to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
    parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
    parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
    parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
    parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
    parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
    parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
    parser.add_argument("--disable_logging", action="store_true", help="Disable wandb logging. Overrides config file if provided.")
    # DAgger related
    parser.add_argument("--expert_path", type=str, help="Path of the expert policy to load. Overrides config file if provided.")
    # Player related
    parser.add_argument("--player_mode", type=str, help="Mode to run the player. Overrides config file if provided.", choices=["soccer"])
    parser.add_argument("--export_policy", action="store_true", help="Export the policy for player usage.")
    parser.add_argument("--policy", type=str, default="train", choices=["train", "teacher", "student"], help="Policy type for eval environment overrides.")
    args = parser.parse_args()

    return args

# Override config file with args
def _update_cfg_from_args(args):
    cfg_file = args.config
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    for arg in vars(args):
        if arg == "policy":
            continue
        if getattr(args, arg) is not None:
            if arg == "num_envs":
                cfg["env"][arg] = getattr(args, arg)
            else:
                cfg["basic"][arg] = getattr(args, arg)

    if args.expert_path is not None:
        cfg["basic"]["expert_path"] = args.expert_path

    if not args.test:
        cfg["viewer"]["record_video"] = False

    if args.disable_logging:
        cfg["runner"]["use_wandb"] = False

    return cfg

def _apply_policy_eval_overrides(cfg, policy):
    if policy == "train":
        return

    cfg["basic"]["task"] = {
        "teacher": "TeacherPolicyEval",
        "student": "StudentPolicyEval",
    }[policy]
    cfg["env"]["early_termination"] = False

def _apply_test_overrides(cfg, args):
    if args.test:
        cfg["env"]["num_envs"] = 1

if __name__ == "__main__":
    args = _get_args()
    cfg = _update_cfg_from_args(args)
    _apply_test_overrides(cfg, args)
    
    if args.export_policy:
        Runner = eval(args.alg)
        runner = Runner(cfg, test=True)
        runner.export_policy("exported_policy.pt")
        exit()
    
    if args.test:
        Runner = eval(args.alg)
        runner = Runner(cfg, test=True)
        if args.player_mode == "soccer":
            runner.play_soccer()
        else:
            runner.play()
    elif args.eval:
        _apply_policy_eval_overrides(cfg, args.policy)
        Runner = eval(args.alg)
        runner = Runner(cfg, test=False)
        runner.evaluate()
    else:
        Runner = eval(args.alg)
        runner = Runner(cfg, test=False)
        runner.train()
