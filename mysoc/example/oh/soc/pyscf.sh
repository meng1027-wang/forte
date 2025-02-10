#!/bin/bash
#
# Sample Slurm job submission.

### Job name
#SBATCH --job-name pyscf
### Job destination
#SBATCH -p amd_512
### Declare job non-rerunnable
#SBATCH --no-requeue
### Output files
#SBATCH -e test_%j.err
#SBATCH -o test_%j.log
### Number of nodes, tasks, cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
### Timeout limits
#SBATCH --time=100:00:00
### Memory limits
#SBATCH --mem=12G

# This job's working directory
echo Working directory is $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
# printenv
echo This jobs runs on the nodes "$SLURM_JOB_NODELIST"
echo This job has allocated $SLURM_NNODES nodes
echo The total number of processes is "$((SLURM_NNODES * SLURM_TASKS_PER_NODE))"
echo The number of cpus per node is "$((SLURM_TASKS_PER_NODE * SLURM_CPUS_PER_TASK))"

# setup environment
export PATH=/public1/home/scg0216/src/gcc-11.2.0-install/bin:$PATH
export LD_LIBRARY_PATH=/public1/home/scg0216/src/gcc-11.2.0-install/lib64:/public1/home/scg0216/src/anaconda3/envs/psi4env/lib:$LD_LIBRARY_PATH
export MKLROOT=/public1/home/scg0216/src/anaconda3/envs/psi4env

PSI4INS=/public1/home/scg0216/src/psi4-install
export PATH=$PSI4INS/bin:$PATH
export PSI4=$PSI4INS/bin/psi4
#export PYTHONPATH=/public1/home/scg0216/src/psi4-build/stage/lib:$PYTHONPATH

export PYTHONPATH=/public1/home/scg0216/src/miniconda3/envs/psi4env/lib/python3.12/site-packages/pyscf:$PYTHONPATH

# Run pyscf with scratch in a temporary directory
PYSCF_SCRATCH=`mktemp -d /tmp/pyscf__XXXXXX`
echo The pyscf scratch dir is $PYSCF_SCRATCH
function finish {
  rm -rf "$PYSCF_SCRATCH"; exit
}
trap finish EXIT

# Run pyscf
export OMP_NUM_THREADS=4
srun -n1 python input.py > output.dat
