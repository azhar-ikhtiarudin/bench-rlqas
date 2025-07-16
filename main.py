import argparse
import subprocess
import copy

agent_choices = ["A2C", "A3C", "DDQN", "DQN", "DQN_PER", "DQN_rank", "Dueling_DQN" , "PPO", "TPPO" ]
task_choices = ['VQE', 'VQSD', 'VQC', 'State_Prep']


def run_main(args):
    print(f"→ Running Task: {args.task} | Agent: {args.agent} | Config: {args.config} | Seed: {args.seed}")
    subprocess.run([
        "python", f"{args.task}/main_{args.agent}.py",
        "--seed", f"{args.seed}",
        "--config", f"{args.task}/configuration_files/{args.agent}/{args.config}",
        "--experiment_name", f"{args.agent}/"
    ])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=task_choices + ["ALL"],
                        help="Choose which main script to run: VQE, VQSD, etc.")
    parser.add_argument("--agent", type=str, required=True, choices=agent_choices + ["ALL"],
                        help="Choose which agent(s) to use.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility")
    parser.add_argument("--config", type=str, default=None, help="Configuration file name")

    args = parser.parse_args()

    # Default config mapping
    default_configs = {
        "VQE": "vanilla_cobyla_H24q0p742_noiseless",
        "VQSD": "vanilla_cobyla_VQSD2q_noiseless_1",
        "VQC": "vanilla_cobyla_3q_VQC",
        "State_Prep": "vanilla_cobyla_H24q0p742_noiseless"
    }

    print("\n### Running Bench-RLQAS with the following setup: ###")
    print("> Task:", args.task)
    print("> Agent:", args.agent)
    print()

    if args.task == "ALL":
        for i, task in enumerate(task_choices):
            if args.agent == "ALL":
                for j, agent in enumerate(agent_choices):
                    task_args = copy.deepcopy(args)
                    task_args.task = task
                    task_args.agent = agent
                    if task_args.config is None:
                        task_args.config = default_configs.get(task, "vanilla_default_config")
                    print(f"Run All: Task-{i}: {task} | Agent-{j}: {agent}")
                    run_main(task_args)
            else:
                task_args = copy.deepcopy(args)
                task_args.task = task
                if task_args.config is None:
                    task_args.config = default_configs.get(task, "vanilla_default_config")
                print(f"Run Task-{i}: {task} | Agent: {task_args.agent}")
                run_main(task_args)
    else:
        if args.agent == "ALL":
            for j, agent in enumerate(agent_choices):
                task_args = copy.deepcopy(args)
                task_args.agent = agent
                if task_args.config is None:
                    task_args.config = default_configs.get(args.task, "vanilla_default_config")
                print(f"Run Task: {args.task} | Agent-{j}: {agent}")
                run_main(task_args)
        else:
            if args.config is None:
                args.config = default_configs.get(args.task, "vanilla_default_config")
            run_main(args)