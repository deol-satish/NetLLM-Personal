import os
import sys
import numpy as np
import torch
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger


PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48
    },
    'llama': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}


def save_model(args, model, save_dir):
    print(f"Saving model with rank {args.rank} to directory {save_dir}")
    if args.rank > 0:
        model.plm.save_pretrained(save_dir)
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    print(f"Loading model from directory {model_dir} with rank {args.rank}")
    if args.rank > 0:
        model.plm.load_adapter(model_dir, adapter_name='default')
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn):
    print(f"Starting adaptation with {args.num_epochs} epochs")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.

    total_train_losses = []
    for epoch in range(args.num_epochs):
        print(f"Training epoch {epoch+1}/{args.num_epochs}")
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        if epoch % args.save_checkpoint_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, str(epoch))
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_model(args, model, checkpoint_dir_epoch)
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        if epoch % args.eval_per_epoch == 0:
            print("Evaluating model...")
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
                                        process_reward_fn=eval_process_reward_fn)
            episodes_return = eval_logs['episodes_return']
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                save_model(args, model, best_model_dir)
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)

    print(f"Saving training losses to {os.path.join(checkpoint_dir, 'train_losses.txt')}")
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn):
    print(f"Testing model, loading from {model_dir}")
    model = load_model(args, model, model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print("Test results:")
    print(results)
    print('Test time:', results['time'], '\nMean reward:', results['mean_reward'])
    print('Results saved at:', result_dir)


def run(args):
    print("Running with the following arguments:")
    pprint(args)

    assert args.plm_type in cfg.plm_types, f"PLM type {args.plm_type} not supported"
    assert args.plm_size in cfg.plm_sizes, f"PLM size {args.plm_size} not supported"
    assert args.exp_pool_path is not None, 'Please specify an experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys(), f"Trace {args.trace} not found in configuration"
    assert args.video in cfg.video_size_dirs.keys(), f"Video {args.video} not found in configuration"

    set_random_seed(args.seed)

    print("Loading traces and setting up environment...")
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time ,all_cooked_bw ,all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    args.trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        args.trace_num = len(all_file_names)
    if args.trace_num == len(all_file_names):
        args.fixed_order = True

    env_settings = {
        'all_cooked_time': all_cooked_time,
        'all_cooked_bw': all_cooked_bw,
        'all_file_names': all_file_names,
        'all_mahimahi_ptrs': all_mahimahi_ptrs,
        'video_size_dir': video_size_dir,
        'fixed': args.fixed_order,
        'trace_num': args.trace_num,
    }
    print("Environment settings:")
    pprint(env_settings)

    print("Loading experience pool...")
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    print("Loading and setting up model...")
    plm, *_ = load_plm(args.plm_type, os.path.join(cfg.plm_dir, args.plm_type, args.plm_size), 
                       device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)

    if args.plm_type != 'llama':
        plm = plm.to(args.device)
    
    if args.rank != -1:
        print(f"Applying PEFT model with rank {args.rank}")
        plm = peft_model(plm, args.plm_type, rank=args.rank)

    assert args.state_feature_dim is not None, 'Please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size, 
                                           max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer)
    
    print("Model setup complete. Directories and paths...")
    train_exp_pool_info = args.exp_pool_path.split('/')[-4:-1]
    train_exp_pool_info = '_'.join(train_exp_pool_info)
    models_dir = os.path.join(cfg.plm_ft_dir, f'{args.plm_type}_{args.plm_size}', train_exp_pool_info + f'_ss_{args.sample_step}', f'rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'\
                              f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    checkpoint_dir = os.path.join(models_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_model_dir = os.path.join(models_dir, 'best')
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    
    result_dir = os.path.join(models_dir, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if args.eval or args.test:
        model_dir = best_model_dir if args.eval else args.test_model_path
        if args.eval:
            print("Evaluating the model...")
            eval_process_reward_fn = lambda rewards, time, return_v: 1. * rewards[-1] if len(rewards) > 0 else 0.
            adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn)
        if args.test:
            print("Testing the model...")
            test_process_reward_fn = lambda rewards, time, return_v: np.mean(rewards)
            test(args, rl_policy, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn)
    else:
        print("Adapting the model...")
        eval_process_reward_fn = lambda rewards, time, return_v: 1. * rewards[-1] if len(rewards) > 0 else 0.
        adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn)
    

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plm-type', type=str, required=True, help='The pretrained language model to use')
    parser.add_argument('--plm-size', type=str, required=True, help='Size of the pretrained language model')
    parser.add_argument('--exp-pool-path', type=str, required=True, help='Path to experience pool')
    parser.add_argument('--trace', type=str, required=True, help='Trace to use for simulation')
    parser.add_argument('--video', type=str, required=True, help='Video size directory')
    parser.add_argument('--state-feature-dim', type=int, help='State feature dimension')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--w', type=int, help='Window size')
    parser.add_argument('--device', type=str, help='Device to use for model')
    parser.add_argument('--device-out', type=str, help='Device to use for output')
    parser.add_argument('--device-mid', type=str, help='Device to use for intermediate layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--rank', type=int, default=-1, help='Rank for low-rank adaptation')
    parser.add_argument('--warmup-steps', type=int, default=0, help='Warmup steps for learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--save-checkpoint-per-epoch', type=int, default=1, help='Save checkpoint every n epochs')
    parser.add_argument('--eval-per-epoch', type=int, default=1, help='Evaluate every n epochs')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--fixed-order', action='store_true', help='Fix order of traces and videos')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale for experience dataset')
    parser.add_argument('--sample-step', type=int, default=1, help='Sample step for experience dataset')
    parser.add_argument('--target-return-scale', type=float, default=1.0, help='Target return scale')
    parser.add_argument('--eval', action='store_true', help='Whether to run evaluation')
    parser.add_argument('--test', action='store_true', help='Whether to run testing')
    parser.add_argument('--test-model-path', type=str, help='Model path to use for testing')

    args = parser.parse_args()
    run(args)
