"""Main tuning script for NoEsis"""

from datetime import datetime
import os
import pprint
import argparse
import sys
import time
import warnings

import datasets
import torch
import torch.distributed as dist
import transformers
import wandb

from NoEsis import logger, util, util_data, util_model, dp_trainer


def run_training(args, model, train_data, test_data=None):
    """Runs training with HuggingFace Accelerated Trainer"""
    logger.info("Starting main loop")

    training_args = transformers.TrainingArguments(
        report_to='wandb',
        output_dir=args.save_dir,
        save_safetensors=False,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='steps',
        save_steps=args.save_freq,
        save_total_limit=2,

        # Evaluation
        do_eval=True,
        eval_strategy='steps',
        eval_steps=min((int(5*args.save_freq), args.eval_freq)),
        use_legacy_prediction_loop=False,
        batch_eval_metrics=True,
        per_device_eval_batch_size=int(args.batch_size_per_replica * 2.),  # Mult by 2 for eval

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.lr_warmup_steps,
        lr_scheduler_type=args.lr_schedule,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        log_level='debug',

        dataloader_drop_last=True,
        dataloader_num_workers=8,

        seed=args.seed,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    privacy_args = dp_trainer.DpArguments(
        disable_dp=args.disable_dp,
        per_sample_max_grad_norm=args.max_gradient_norm,
        target_delta=float(1e-6),
        target_epsilon=args.target_epsilon,
        noise_multiplier=args.noise_multiplier,
        grad_sample_mode=args.grad_sample_mode,
    )

    # Attach settings to model.config, which WandB will use for labels/legend
    model.config.datasets = util.strip_datasets(args.datasets)
    model.config.disable_dp = privacy_args.disable_dp
    model.config.finetune_second = args.finetune_second
    model.config.freeze_common = args.freeze_common
    model.config.freeze_domain = args.freeze_domain
    model.config.max_gradient_norm = privacy_args.per_sample_max_grad_norm
    model.config.noise_multiplier = privacy_args.noise_multiplier
    model.config.num_day = (datetime.now() - datetime(2024, 10, 1)).days
    model.config.num_gpu = torch.cuda.device_count()
    model.config.num_node = int(os.environ.get('NNODENUM', 1))
    model.config.num_prompt_tokens = max((args.num_prompt_tokens, model.config.num_prompt_tokens))
    model.config.num_prefix_tokens = max((args.num_prefix_tokens, model.config.num_prefix_tokens))
    model.config.target_epsilon = privacy_args.target_epsilon
    model.vast_label = str(os.environ.get('VAST_CONTAINERLABEL', ''))

    model.train()

    trainer = dp_trainer.NoEsisTrainer(
        args=training_args,
        model=model,
        data_collator=util_data.custom_collate_fn,
        train_dataset=train_data,
        eval_dataset=test_data,
        privacy_args=privacy_args)

    # Check for existing checkpoints
    last_checkpoint = None
    checkpoint_prefix = "checkpoint-"
    if args.continue_from_checkpoint:
        last_checkpoint = args.continue_from_checkpoint
    elif os.path.isdir(args.save_dir):
        checkpoints = [
            d for d in os.listdir(args.save_dir)
            if os.path.isdir(os.path.join(args.save_dir, d)) and d.startswith(checkpoint_prefix)]

        if checkpoints:
            logger.info(f"Found {len(checkpoints)} checkpoints: {','.join(checkpoints)}")
            # Sort checkpoints by the integer value after 'checkpoint-' and select the latest one
            checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(checkpoint_prefix, '')))
            last_checkpoint = os.path.join(args.save_dir, checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Start training, resuming if a checkpoint is available
    # If ValueError: loaded state dict contains a parameter group that doesn't match..., then
    # check that args.num_experts is the same as the model being loaded
    print(f"Number of trainable tensors: {len([p for p in model.parameters() if p.requires_grad])}")
    print(f"Log freq: {args.log_freq}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if args.local_rank <= 0:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print("\n", flush=True)
        print(
            f'  ==> Finish training and save to {final_checkpoint_dir} '
            f'on Node {os.environ.get("NNODENUM")}/{os.environ.get("VAST_CONTAINERLABEL")}',
            flush=True)


def load_tokenize_data(args, split='train'):
    """Loads the tokenized data from cache or from scratch"""
    # Load and tokenize data
    cachedir = os.path.join(args.cache_data, util.strip_datasets(args.datasets), split)
    cachedir = os.path.join(cachedir, "dedup" if args.use_dedup else "original")
    if os.path.exists(cachedir):
        train_data = datasets.load_from_disk(cachedir)
        logger.info(f'  ==> Loaded {len(train_data)} samples from cache at {cachedir} '
                    f'on rank {args.local_rank}')
        return train_data

    train_data = util_data.provision_dataset(args, split=split)
    logger.info(f'  ==> Loaded {len(train_data)} samples')

    train_data.save_to_disk(cachedir)
    logger.info(f'  ==> Saved to {cachedir} on rank {args.local_rank}')
    return train_data


def main(args):
    """Loads the model and the data and starts training"""
    argsdict = vars(args)
    logger.info(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:  # pylint: disable=unspecified-encoding
        f.write(pprint.pformat(argsdict))

    warnings.filterwarnings("ignore", ".*UserWarning.*")
    warnings.filterwarnings("ignore", ".*FutureWarning.*")
    warnings.filterwarnings("ignore", ".*DeprecationWarning.*")

    if args.local_rank > 0:
        torch.distributed.barrier()

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached,
    # load it from there.
    train_data = load_tokenize_data(args, split='train')
    test_data = load_tokenize_data(args, split='test')
    # Randomly select 10% of the test_data for a smaller test set
    test_data = test_data.shuffle(seed=args.seed).select(range(int(0.1 * len(test_data))))

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Load model from `args.load`
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.load)
    model = util_model.load_upstream_model(args)

    # use only the decoder part of the model
    for layer in model.decoder.block:
        layer.layer[0].SelfAttention.cross_attention = False

    print(f"CUDA VISIBLE DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Python process ID: {os.getpid()} / {os.getppid()}")
    print(f"Local rank: {args.local_rank} on Node {os.environ.get('NNODENUM')}")
    run_training(args, model, train_data, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--datasets', default='java,python', type=str, help='Comma separated')
    parser.add_argument('--max-target-len', default=512, type=int)
    parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    parser.add_argument('--parallel-mode', default='not_distributed', type=str)

    # Training
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--disable-dp', default=False, action='store_true')
    parser.add_argument('--epochs', default=3, type=float)
    parser.add_argument('--expert-layer-start', default=-1, type=int)  # Which layers to expertize
    parser.add_argument('--finetune-second', default=-1, type=int)  # Make experts on existing
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--freeze-backbone', default=False, action='store_true')
    parser.add_argument('--freeze-common', default=False, action='store_true')
    parser.add_argument('--freeze-domain', default=False, action='store_true')
    parser.add_argument('--grad-acc-steps', default=1, type=int)
    parser.add_argument('--grad-sample-mode', default='hooks',
                        type=str, choices=['ew', 'hooks', 'ghost', 'functorch'])
    parser.add_argument('--lora-alpha', default=0.1, type=float)  # negative number disables lora
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr-schedule', default='linear', type=str)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--max-gradient-norm', default=1., type=float)
    parser.add_argument('--noise-multiplier', default=-1., type=float)
    parser.add_argument('--num-experts', default=-1, type=int)  # Number of experts per layer
    parser.add_argument('--num-prompt-tokens', default=-1, type=int)  # Number of learnable tokens
    parser.add_argument('--num-prefix-tokens', default=-1, type=int)  # Number of learnable tokens
    parser.add_argument('--rank-common', default=-1, type=int)  # negative number disables lora
    parser.add_argument('--rank-domain', default=-1, type=int)  # negative number disables lora
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--target-epsilon', default=-1., type=float)
    parser.add_argument('--use-dedup', default=False, action='store_true')
    parser.add_argument('--weight-decay', default=0.05, type=float)

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument(
        '--ckpt-dir', default="/root/code/NoEsis/NoEsis/saved_models/checkpoint_upstream", type=str)
    parser.add_argument('--continue-from-checkpoint', default=None, type=str)
    parser.add_argument('--log-freq', default=50, type=int)
    parser.add_argument('--save-freq', default=1000, type=int)  # Training is generally .8 s/it
    parser.add_argument('--eval-freq', default=99999, type=int)

    args = parser.parse_args()

    if args.local_rank < 0:
        if int(os.environ.get('LOCAL_RANK', -1)) > -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK'))

    if args.local_rank >= 0:
        dist.init_process_group(backend="nccl")

    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')  # pylint: disable=consider-using-with
        # Hacky solution. Prevent race condition for whoever gets to create the logging file first
        time.sleep(2 + 3*int(args.local_rank))

    if args.local_rank > 0:
        # Suppress logging from processes other than rank 0
        wandb.setup().settings.show_info = False
        wandb.setup().settings.show_warnings = False
        wandb.setup().settings.silent = True
        os.environ['WANDB_CONSOLE'] = 'off'
        # Remove stream handler from logger to prevent printing to stdout
        for handler in logger.handlers:
            logger.removeHandler(handler)

    with wandb.init(project='noesis', entity='brave-research'):
        config_wandb = wandb.config
        print(f"Overwrites: {config_wandb.as_dict()}")

        for key, value in config_wandb.as_dict().items():
            assert hasattr(args, key), f"Key {key} not found in args"
            setattr(args, key, value)
        print(f"Args after overwrite from WandB: {args}")

        util.make_git_log()

        savedir = os.path.join(
            args.save_dir,
            (f'{util.strip_datasets(args.datasets)}_num{args.num_experts}_'
             f'seed{args.seed}_{str(wandb.run.id)}'))
        os.makedirs(savedir, exist_ok=True)
        args.save_dir = savedir
        main(args)
