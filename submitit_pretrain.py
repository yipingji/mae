# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    
    parser.add_argument('--skipless', action='store_true',help='if use skipless training')
    parser.add_argument('--mimetic', default=None, type=float, nargs=2, help='mimetic init')
    parser.add_argument('--W_v', default=1.0, type=float, help='coefficient for Value')
    parser.add_argument('--W_p', default=1.0, type=float, help='coefficient for Projection')
        
    return parser.parse_args()


def get_shared_folder() -> Path:
    if Path("/scratch3/ji016/project/2025/dino/output").is_dir():
        p = Path(f"/scratch3/ji016/project/2025/dino/output")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.mimetic is not None:
        args.output_dir =  f"output/dino_{args.arch}_ep{args.epochs}_bs{args.batch_size_per_gpu}_opt{args.optimizer}_lr{args.lr}_skipless{args.skipless}_svdortho_mimetic{args.mimetic[0]}_{args.mimetic[1]}_Wv{args.W_v}_Wp{args.W_p}"
    else:
        args.output_dir =  f"output/dino_{args.arch}_ep{args.epochs}_bs{args.batch_size_per_gpu}_opt{args.optimizer}_lr{args.lr}_skipless{args.skipless}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    if args.timeout <=120:
        args.partition = 'h2gpu'
    elif args.timeout <= 1440:
        args.partition = 'h24gpu'
    kwargs['slurm_account'] = 'OD-236362'

    executor.update_parameters(
        mem_gb = 40* args.ngpus,  
        nodes=args.nodes,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        cpus_per_task=16,
        **kwargs,
    )


    executor.update_parameters(name="mae")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()
