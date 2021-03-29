"""Write helper script to submit job to slurm
"""

import os, sys
import subprocess
import platform
import argparse
import datetime
from colorama import init, Fore

init(autoreset=True)

SLURM_CMD = """#!/bin/bash

# set a job name
#SBATCH --job-name={job_name}
#################

# a file for job output, you can check job progress
#SBATCH --output={output}
#################

# a file for errors
#SBATCH --error={error}
#################

# time needed for job
#SBATCH --time={time}
#################

# gpus per node
#SBATCH --gres=gpu:{num_gpus}
#################

# number of requested nodes
#SBATCH --nodes={num_nodes}
#################

# slurm will send a signal this far out before it kills the job
{auto_submit}
#################


# #task per node
#SBATCH --ntasks-per-node={ntasks_per_node}
#################

# #cpu per task/gpu
#SBATCH --cpus-per-task={cpus_per_task}
#################

# memory per cpu
#SBATCH --mem-per-cpu={mem_per_cpu}
#################

# extra stuff
{extra}


export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

{module}

srun {main_cmd}
"""


def slurm_parser(parent=None):
    parser = argparse.ArgumentParser(parents=[parent],
                                     conflict_handler='resolve')
    parser.add_argument('-n',
                        '--num_nodes',
                        default=1,
                        type=int,
                        help='number of nodes')
    parser.add_argument('-j',
                        '--job_name',
                        default=None,
                        help='job name and log name prefix')
    parser.add_argument('-t',
                        '--time',
                        default='6',
                        type=str,
                        help='hours or (hh:mm:ss)')
    parser.add_argument("--num_gpus", "-g", type=int, default=1)
    parser.add_argument('--init_dep', type=str, default=None)
    parser.add_argument(
        "--auto_resubmit",
        "-a",
        action='store_true',
        help=
        "whether to auto submit wall time ('None' means no wall time submission)"
    )

    parser.add_argument("--slurm_log_root", "-l", type=str, default="results")
    # parser.add_argument("--memory", type=int, default=500000)
    parser.add_argument("--cpus_per_task", type=int, default=6)
    parser.add_argument("--mem_per_cpu", type=str, default='10000')
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--from_slurm", action="store_true")

    return parser


def get_max_trial_version(path):
    files = os.listdir(path)
    version_files = [f for f in files if 'trial_' in f]
    if len(version_files) > 0:
        # regex out everything except file version for ve
        versions = [int(f_name.split('_')[1]) for f_name in version_files]
        max_version = max(versions)
        return max_version + 1
    else:
        return 0


def layout_path(params):
    # format the logging folder path
    slurm_out_path = os.path.join(params.slurm_log_root, params.job_name)

    # when err logging is enabled, build add the err logging folder
    err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
    if not os.path.exists(err_path):
        os.makedirs(err_path)

    # when out logging is enabled, build add the out logging folder
    out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # place where slurm files log to
    slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')
    if not os.path.exists(slurm_files_log_path):
        os.makedirs(slurm_files_log_path)
    return out_path, err_path, slurm_files_log_path


def run_cluster(parser, fn_main, lt_system):
    params = parser.parse_args()

    if params.job_name is None:
        params.job_name = params.model_name
    else:
        params.job_name = params.job_name

    if params.from_slurm:
        fn_main(params, lt_system)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        extra = ""
        params.slurm_log_root = os.path.join(os.environ['HOME'],
                                             params.slurm_log_root)
        out_path, err_path, slurm_files_log_path = layout_path(params)
        trial_version = get_max_trial_version(out_path)
        if not ':' in params.time:
            params.time = f"{int(params.time):02d}:00:00"
        if params.auto_resubmit is False:
            auto_walltime = ""
        else:
            auto_walltime = f"#SBATCH --signal=USR1@150"

        arch = platform.uname().processor

        if params.partition is not None:
            extra = f"#SBATCH --partition {params.partition} \n"

        if arch == 'x86_64':
            loaded_module = "module load gcc cuda "
        elif arch == 'ppc64le':
            loaded_module = "module load spectrum-mpi"
        else:
            loaded_module = ''

        loaded_module = ''  # FIXME
        extra = extra + "conda activate torch1.7 \n"

        python_cmd = " ".join(sys.argv)
        full_command = f"python {python_cmd} --from_slurm "
        cmd_to_sbatch = SLURM_CMD.format(
            job_name=params.job_name,
            output=os.path.join(
                out_path,
                f'trial_{trial_version}_{timestamp}_slurm_output_%j.out'),
            error=os.path.join(
                err_path,
                f'trial_{trial_version}_{timestamp}_slurm_err_%j.out'),
            time=params.time,
            num_gpus=params.num_gpus,
            num_nodes=params.num_nodes,
            auto_submit=auto_walltime,
            email="",
            ntasks_per_node=params.num_gpus,
            cpus_per_task=params.cpus_per_task,
            mem_per_cpu=params.mem_per_cpu,
            extra=extra,
            module=loaded_module,
            main_cmd=full_command,
        )
        print(Fore.LIGHTBLUE_EX + cmd_to_sbatch)
        script = "{}/{}.sh".format(slurm_files_log_path, params.job_name)
        print("Generate sbatch script at {}".format(script))
        with open(script, 'w') as f:
            print(cmd_to_sbatch, file=f, flush=True)

        p = subprocess.Popen(['sbatch', script],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, _ = p.communicate()
        stdout = stdout.decode("utf-8")
        job_id = stdout.split(" ")[-1]
        print(Fore.CYAN + f"Job {job_id.strip()} is submitted.")
        print(Fore.GREEN + "-" * 60, "\n\n\n")
