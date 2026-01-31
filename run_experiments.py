"""
Experiment Runner Script
Runs all implemented algorithms (SB3 baselines + TOP variants) on HalfCheetah-v5 and Hopper-v5.

Usage:
  # Run all experiments for a specific seed:
  python run_experiments.py --seed 1

  # Run all experiments sequentially:
  python run_experiments.py

  # Run a specific algorithm:
  python run_experiments.py --algorithm sb3_sac

  # Run a specific environment:
  python run_experiments.py --env HalfCheetah-v5

  # Run specific experiment by index (useful for parallel execution on different machines):
  python run_experiments.py --experiment_id 0
  
  # List all experiments:
  python run_experiments.py --list
"""

import subprocess
import argparse
import sys
from typing import List, Dict
import os


# Define all experiments
ALGORITHMS = {
    # === CleanRL Baselines (for comparison, but not recommended - use SB3 instead) ===
    'sac_cleanrl': {
        'script': 'sac.py',
        'description': 'SAC (CleanRL - not tuned)',
        'args': {
            '--total_timesteps': '1000000'
        },
        'env_arg': '--env_id'
    },
    'td3_cleanrl': {
        'script': 'td3.py',
        'description': 'TD3 (CleanRL - not tuned)',
        'args': {
            '--total_timesteps': '1000000'
        },
        'env_arg': '--env_id'
    },
    'tqc_cleanrl': {
        'script': 'tqc.py',
        'description': 'TQC (CleanRL - not tuned)',
        'args': {
            '--n_quantiles': '25',
            '--n_critics': '5',
            '--top_quantiles_to_drop': '2',
            '--total_timesteps': '1000000'
        },
        'env_arg': '--env_id'
    },
    
    # === Stable Baselines3 Baselines (RECOMMENDED - properly tuned) ===
    'td3': {
        'script': 'sb3_td3.py',
        'description': 'TD3 (SB3 with RL Zoo hyperparameters)',
        'args': {
            '--total_timesteps': '1000000',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env_id'
    },
    'sac': {
        'script': 'sb3_sac.py',
        'description': 'SAC (SB3 with RL Zoo hyperparameters)',
        'args': {
            '--total_timesteps': '1000000',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env_id'
    },
    'tqc': {
        'script': 'sb3_tqc.py',
        'description': 'TQC (SB3 with RL Zoo hyperparameters)',
        'args': {
            '--total_timesteps': '1000000',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env_id'
    },
    
    # === TOP Variants (SAC-based with distributional critics) ===
    'top_sac': {
        'script': 'mujoco/train_top_sac.py',
        'description': 'TOP-SAC with bandit-controlled beta (state-based SAC)',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--bandit_lr': '0.1',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_learnable': {
        'script': 'mujoco/train_top_agent.py',
        'description': 'TOP-SAC with learnable beta parameter',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--bandit_lr': '0.1',
            '--learnable_beta': '',
            '--beta': '0',
            '--beta_lr': '3e-4',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_learnable_beta_neg1': {
        'script': 'mujoco/train_top_agent.py',
        'description': 'TOP-SAC with learnable beta (init=-1)',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--bandit_lr': '0.1',
            '--learnable_beta': '',
            '--beta': '-1',
            '--beta_lr': '3e-4',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_optimist': {
        'script': 'mujoco/train_top_agent.py',
        'description': 'TOP-SAC with fixed optimistic beta=0',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--beta': '0',
            '--fixed_beta': '',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_pessimist': {
        'script': 'mujoco/train_top_agent.py',
        'description': 'TOP-SAC with fixed pessimistic beta=-1',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--beta': '-1',
            '--fixed_beta': '',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    
    # === Classical TOP Variant ===
    'top': {
        'script': 'mujoco/train_top_agent.py',
        'description': 'Classical TOP with bandit-controlled optimism',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--bandit_lr': '0.1',
            '--beta': '0.0',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    
    # === TOP-TQC Variants ===
    'top_tqc': {
        'script': 'mujoco/train_top_tqc.py',
        'description': 'TOP-TQC hybrid (TOP + TQC truncation)',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--n_critics': '5',
            '--top_quantiles_to_drop': '4',
            '--bandit_lr': '0.1',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_tqc_learnable': {
        'script': 'mujoco/train_top_tqc.py',
        'description': 'TOP-TQC with learnable beta parameter',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--n_critics': '5',
            '--top_quantiles_to_drop': '4',
            '--bandit_lr': '0.1',
            '--learnable_beta': '',
            '--beta': '0',
            '--beta_lr': '3e-4',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_tqc_learnable_beta_neg1': {
        'script': 'mujoco/train_top_tqc.py',
        'description': 'TOP-TQC with learnable beta (init=-1)',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--n_critics': '5',
            '--top_quantiles_to_drop': '4',
            '--bandit_lr': '0.1',
            '--learnable_beta': '',
            '--beta': '-1',
            '--beta_lr': '3e-4',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
    'top_tqc_learnable_quantiles': {
        'script': 'mujoco/train_top_tqc_learnable_drop.py',
        'description': 'TOP-TQC with learnable quantile dropping',
        'args': {
            '--train_steps': '1000000',
            '--n_quantiles': '50',
            '--n_critics': '5',
            '--top_quantiles_to_drop': '4',
            '--beta': '0.0',
            '--bandit_lr': '0.1',
            '--learnable_drop': '',
            '--drop_lr': '3e-4',
            '--log_interval': '10000',
            '--n_evals': '5'
        },
        'env_arg': '--env'
    },
}

ENVIRONMENTS = ['HalfCheetah-v5', 'Hopper-v5']
SEEDS = [4, 5]


def generate_experiments() -> List[Dict]:
    """Generate all experiment configurations"""
    experiments = []
    exp_id = 0
    
    for algo_name, algo_config in ALGORITHMS.items():
        for env in ENVIRONMENTS:
            for seed in SEEDS:
                experiment = {
                    'id': exp_id,
                    'algorithm': algo_name,
                    'env': env,
                    'seed': seed,
                    'script': algo_config['script'],
                    'description': algo_config['description'],
                    'args': algo_config['args'].copy(),
                    'env_arg': algo_config.get('env_arg', '--env')
                }
                experiments.append(experiment)
                exp_id += 1
    
    return experiments


def build_command(experiment: Dict) -> List[str]:
    """Build the command to run an experiment"""
    cmd = [sys.executable, experiment['script']]
    
    # Add environment (use the correct argument name for this algorithm)
    env_arg = experiment.get('env_arg', '--env')
    cmd.extend([env_arg, experiment['env']])
    
    # Add seed
    cmd.extend(['--seed', str(experiment['seed'])])
    
    # Add standardized save directory if provided by caller
    # The caller (main) will set 'save_dir' in the experiment dict before calling this
    save_dir = experiment.get('save_dir')
    # Only add --save_dir flag; do NOT pass --log_dir to avoid argparse errors
    existing_args = set(experiment.get('args', {}).keys())
    if save_dir and not any(k in existing_args for k in ['--save_dir', '--outdir', '--output_dir']):
        cmd.extend(['--save_dir', save_dir])
    # Do not forward the top-level --algorithm argument to target scripts.
    # Forwarding caused target scripts (e.g., sb3_*.py) to receive unknown
    # arguments and crash. If a script needs the algorithm name, add it
    # explicitly to that algorithm's `args` in the ALGORITHMS dict.

    # Add algorithm-specific arguments
    for arg_name, arg_value in experiment['args'].items():
        if arg_value == '':
            # Flag argument (no value)
            cmd.append(arg_name)
        else:
            cmd.extend([arg_name, arg_value])
    
    return cmd


def run_experiment(experiment: Dict, dry_run: bool = False) -> int:
    """Run a single experiment"""
    cmd = build_command(experiment)
    
    print(f"\n{'='*80}")
    print(f"Experiment ID: {experiment['id']}")
    print(f"Algorithm: {experiment['algorithm']} ({experiment['description']})")
    print(f"Environment: {experiment['env']}")
    print(f"Seed: {experiment['seed']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN - Not executing\n")
        return 0
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment {experiment['id']} completed successfully\n")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {experiment['id']} failed with exit code {e.returncode}\n")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n✗ Experiment {experiment['id']} interrupted by user\n")
        return 1


def list_experiments(experiments: List[Dict], algorithm: str = None, env: str = None, seed: int = None):
    """List all experiments"""
    filtered = experiments
    
    if algorithm:
        filtered = [e for e in filtered if e['algorithm'] == algorithm]
    if env:
        filtered = [e for e in filtered if e['env'] == env]
    if seed is not None:
        filtered = [e for e in filtered if e['seed'] == seed]
    
    print(f"\nTotal experiments: {len(filtered)}\n")
    print(f"{'ID':<4} {'Algorithm':<20} {'Environment':<20} {'Seed':<6}")
    print("-" * 80)
    
    for exp in filtered:
        print(f"{exp['id']:<4} {exp['algorithm']:<20} {exp['env']:<20} {exp['seed']:<6}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run reinforcement learning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--algorithm', type=str, choices=list(ALGORITHMS.keys()),
                       help='Run only this algorithm')
    parser.add_argument('--env', type=str, choices=ENVIRONMENTS,
                       help='Run only this environment')
    parser.add_argument('--seed', type=int, choices=SEEDS,
                       help='Run only this seed')
    parser.add_argument('--experiment_id', type=int,
                       help='Run specific experiment by ID (useful for parallel execution)')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments and exit')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be run without actually running')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start from this experiment ID (useful for resuming)')
    parser.add_argument('--base_save_dir', type=str, default='runs',
                       help='Base directory under which per-experiment folders will be created (default: runs)')
    
    args = parser.parse_args()
    
    # Generate all experiments
    experiments = generate_experiments()
    
    # List mode
    if args.list:
        list_experiments(experiments, args.algorithm, args.env, args.seed)
        return
    
    # Filter experiments based on arguments
    experiments_to_run = experiments
    
    if args.experiment_id is not None:
        # Run specific experiment
        if args.experiment_id < 0 or args.experiment_id >= len(experiments):
            print(f"Error: experiment_id must be between 0 and {len(experiments)-1}")
            sys.exit(1)
        experiments_to_run = [experiments[args.experiment_id]]
    else:
        # Filter by algorithm, env, seed
        if args.algorithm:
            experiments_to_run = [e for e in experiments_to_run if e['algorithm'] == args.algorithm]
        if args.env:
            experiments_to_run = [e for e in experiments_to_run if e['env'] == args.env]
        if args.seed is not None:
            experiments_to_run = [e for e in experiments_to_run if e['seed'] == args.seed]
        
        # Apply start_from filter
        experiments_to_run = [e for e in experiments_to_run if e['id'] >= args.start_from]
    
    if not experiments_to_run:
        print("No experiments match the specified criteria.")
        return
    
    print(f"\nWill run {len(experiments_to_run)} experiment(s)\n")
    
    # Run experiments
    failed = []
    for i, experiment in enumerate(experiments_to_run):
        # prepare base save directory per algorithm and environment (no seed/id)
        base = args.base_save_dir
        # per-environment save dir; training scripts will create per-seed
        # subfolders under this path (seed_<seed>/id_<ts>)
        save_dir = os.path.join(base, experiment['algorithm'], experiment['env'])
        os.makedirs(save_dir, exist_ok=True)
        experiment['save_dir'] = save_dir

        print(f"\nProgress: {i+1}/{len(experiments_to_run)}")
        exit_code = run_experiment(experiment, args.dry_run)
        if exit_code != 0:
            failed.append(experiment['id'])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"Successful: {len(experiments_to_run) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed experiment IDs: {failed}")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")


if __name__ == '__main__':
    main()
