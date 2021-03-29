import pytorch_lightning as pl
import os
import argparse
from configs import config_parser
from pytorch_lightning.utilities.distributed import rank_zero_only
from importlib import import_module
from test_tube import HyperOptArgumentParser, SlurmCluster
import pytorch_lightning as pl
import os
import platform
import copy


def get_parser(parent=None):
    if parent is not None:
        parser = argparse.ArgumentParser(parents=[parent],
                                         conflict_handler='resolve')
    else:
        parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = config_parser(parent=parser)

    args = parser.parse_known_args()[0]
    lt_system = load_system(args.system)

    # add model specific parser
    parser = lt_system.add_model_specific_args(parser)
    return parser, lt_system


def config_init(params):
    if params.seed >= 0:
        pl.seed_everything(params.seed)

    if params.test:
        params.mode = 'test'

    if params.print_freq != -1:
        params.log_every_n_steps = params.print_freq
    else:
        params.log_every_n_steps = int(1e10)

    if isinstance(params.val_dataset, list) and len(params.val_dataset) == 1:
        params.val_dataset = params.val_dataset[0]

    params.base_model_name = copy.deepcopy(params.model_name)

    params.model_name = f'{params.model_name}_{params.model}'

    if params.mode == 'test' and not params.resume:
        params.model_name += ("_" + str(params.val_dataset))
        # params.model_name += ("_" + params.fine_tune_type)

    if params.suffix is not None:
        params.model_name += params.suffix

    # uname = os.uname()[1]
    # if 'dcs' in uname or 'npl' in uname:
    #     uname = 'cci'
    # elif 'dcc' in uname:
    #     uname = 'dcc'
    uname = 'cci'

    params.save_path = os.path.join('ckpt', uname, params.model_name)
    if params.mode != 'test' and not os.path.isdir(params.save_path):
        if rank_zero_only.rank == 0:
            os.makedirs(params.save_path, exist_ok=True)

    params.log_path = 'logs'

    if params.mode == 'test':
        params.print_freq = 5

    if params.resume and os.path.exists(
            os.path.join(params.save_path, 'last.ckpt')):
        params.resume_from_checkpoint = os.path.join(params.save_path,
                                                     'last.ckpt')
        if params.test: params.ckpt = params.resume_from_checkpoint

    return params


def refine_args(params):
    def fn(name):
        # convert comma-separated dataset to list
        if hasattr(params, name) and isinstance(getattr(params, name), str):
            val = getattr(params, name)
            _split = val.split(',')
            if len(_split) > 1:
                setattr(params, name, _split)

    fn('dataset')
    fn('val_dataset')

    # set backend properly
    if params.gpus == '-1':
        params.gpus = -1
    else:
        _split = params.gpus.split(',')
        if len(_split) == 1:
            params.gpus = int(params.gpus)
        else:
            list_of_gpu = [int(k) for k in _split if k.isdigit()]
            params.gpus = list_of_gpu
    if ((isinstance(params.gpus, list) and len(params.gpus) > 1) or
        (isinstance(params.gpus, int) and params.gpus > 1)) or (params.gpus
                                                                == -1):
        params.distributed_backend = 'ddp'

    # # : for now, don't do DDP in test, there's a bug in pl
    # if params.mode == 'test':
    #     params.distributed_backend = None
    #     params.gpus = params.gpus[0] if (isinstance(params.gpus, list)) else 1
    return params


def load_system(system_name):
    if system_name is not None:
        module = import_module(f"system.system_{system_name}", __package__)
        lt_system = module.LightningSystem
    else:
        raise ValueError(f"should provide system!!")
    return lt_system


def testtube_parser(parent=None, strategy='random_search'):
    parser = HyperOptArgumentParser(parents=[parent],
                                    conflict_handler='resolve',
                                    strategy=strategy)
    parser.add_argument('-n',
                        '--num_nodes',
                        default=1,
                        type=int,
                        help='number of nodes')
    parser.add_argument('-d',
                        '--num_deps',
                        default=1,
                        type=int,
                        help='number of consecutive jobs')
    parser.add_argument('-j',
                        '--job_name',
                        default=None,
                        help='job name and log name prefix')
    parser.add_argument('-t',
                        '--time',
                        default='6',
                        type=str,
                        help='hours or (hh:mm:ss)')
    parser.add_argument("--n_gpus", "-g", type=int, default=6)
    parser.add_argument('--init_dep', type=str, default=None)
    parser.add_argument(
        "--auto_resubmit",
        "-a",
        action='store_true',
        help=
        "whether to auto submit wall time ('None' means no wall time submission)"
    )

    parser.add_argument("--slurm_log_root", "-l", type=str, default="results")
    parser.add_argument("--nb_trials", type=int, default=1)
    # parser.add_argument("--memory", type=int, default=500000)
    parser.add_argument("--cpus_per_task", type=int, default=6)
    parser.add_argument("--mem_per_cpu", type=str, default='10000')
    parser.add_argument("--hyper_opt", action="store_true")
    parser.add_argument("--partition", type=str, default=None)
    return parser


def run_cluster(parser, fn_main, lt_system):
    params = parser.parse_args()

    if params.system_mode == "3d" and "3d" not in params.model_name:
        params.model_name += "_3d"

    if not ':' in params.time:
        params.time = f"{int(params.time):02d}:00:00"

    arch = platform.uname().processor

    loaded_module = ''
    partition = params.partition
    # if partition is None:
    if arch == 'x86_64':
        partition = 'npl'
    elif arch == 'ppc64le':
        partition = 'dcs,rpi'

    if partition == 'npl':
        loaded_module = "module load gcc cuda openmpi"
    else:
        loaded_module = "module load spectrum-mpi"

    log_path = os.path.join(os.environ['HOME'], params.slurm_log_root)

    cluster = SlurmCluster(hyperparam_optimizer=params,
                           log_path=log_path,
                           python_cmd="python")

    # cluster.notify_job_status(email='',
    #                           on_fail=True,
    #                           on_done=False)
    # configure cluster
    cluster.per_experiment_nb_gpus = params.n_gpus
    cluster.per_experiment_nb_nodes = params.num_nodes
    cluster.per_experiment_nb_cpus = 0  # disable this option
    cluster.job_time = params.time
    cluster.minutes_to_checkpoint_before_walltime = 2  # 2 min walltime
    cluster.memory_mb_per_node = int(params.n_gpus) * int(
        params.cpus_per_task) * int(params.mem_per_cpu)

    if params.partition is not None:
        cluster.add_slurm_cmd('partition',
                              value=params.partition,
                              comment='cluster partition name')
    cluster.add_slurm_cmd('ntasks-per-node',
                          value=params.n_gpus,
                          comment='#task per node')
    cluster.add_slurm_cmd('cpus-per-task',
                          value=params.cpus_per_task,
                          comment='#cpu per task/gpu')
    cluster.add_slurm_cmd('mem-per-cpu',
                          value=params.mem_per_cpu,
                          comment="memory per cpu")

    # cluster.memory_mb_per_node = params.memory  # disable this option

    cluster.add_command('export PYTHONFAULTHANDLER=1')
    # cluster.add_command('export NCCL_DEBUG=INFO')
    cluster.add_command(loaded_module)

    # Master address for multi-node training
    cluster.add_command(
        "export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\\n' ' ')"
    )
    cluster.add_command("export SLURM_NODELIST=$SLURM_JOB_NODELIST")
    cluster.add_command(
        "slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)")
    cluster.add_command(
        "export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)")

    if params.job_name is None:
        job_name = params.model_name
    else:
        job_name = params.job_name

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute
        lambda par, _optimizer: fn_main(par, lt_system, _optimizer),
        # Number of hyperparameter combinations to search:
        nb_trials=params.nb_trials,
        enable_auto_resubmit=params.auto_resubmit,
        # This is what will display in the slurm queue:
        job_name=job_name)
