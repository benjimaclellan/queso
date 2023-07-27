import copy
import os

from queso.io import IO
from queso.configs import Configuration


if __name__ == "__main__":

    base = Configuration()

    folders = {}
    for n in (2, 4):
        config = copy.deepcopy(base)
        config.n = n

        folder = f"2023-07-27_n={config.n}_k={config.k}"
        jobname = f"n{config.n}k{config.k}"

        io = IO(folder=folder)
        io.path.mkdir(parents=True, exist_ok=True)
        config.to_yaml(io.path.joinpath('config.yaml'))
        print(io.path.name)

        path_str = """
        export PYTHONPATH="/Users/benjamin/Library/CloudStorage/OneDrive-UniversityofWaterloo/Desktop/1 - Projects/Quantum Intelligence Lab/queso:${PYTHONPATH}"
        """

        preamble_str = """
        module purge
        module load python/3.9 scipy-stack
        module load cuda/11.4
        module load cudnn/8.2.0
        source ~/bash-profiles/queso
        source ~/venv/queso_venv/bin/activate

        cd ~/projects/def-rgmelko/bmaclell/queso
        """

        sbatch_opt = " ".join([
            f"--job-name={jobname}.job",
            f"--account=def-rgmelko",
            f"--job-name={jobname}.job",
            f"--time=0:05:00",
            f"--mem=12000",
            # f"--gpus-per-node=1",
            f"--output=.out/{jobname}.out",
            f"--error=.out/{jobname}.err",
        ])

        python_str = f"python experiments/train.py --{folder}"
        sbatch_str = f"{sbatch_str} {python_str}"

        command = (
                path_str
                # + preamble_str
                + python_str
                # + sbatch_str
        )
        # os.system(command)
        print(command)
