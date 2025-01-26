#!/bin/bash
#
# Sample Slurm job submission.

### Job name
#SBATCH --job-name cu
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
#SBATCH --cpus-per-task=25
### Timeout limits
#SBATCH --time=100:00:00
### Memory limits
#SBATCH --mem=100G

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
source /public1/soft/modules/module.sh
module purge
module load gcc/12.2
export MKLROOT=/public1/home/scg0216/src/anaconda3/envs/psi4env

PSI4INS=/public1/home/scg0216/src/psi4-install
export PATH=$PSI4INS/bin:$PATH
export PSI4=$PSI4INS/bin/psi4
export PSIDATADIR=/public1/home/scg0216/src/psi4/psi4/share/psi4
export PYTHONPATH=/public1/home/scg0216/src/wm_forte_new/:/public1/home/scg0216/src/psi4-build/stage/lib:$PYTHONPATH

# Run psi4 with scratch in a temporary directory
PSI4_SCRATCH=`mktemp -d /tmp/psi4__XXXXXX`
echo The psi4 scratch dir is $PSI4_SCRATCH
function finish {
  rm -rf "$PSI4_SCRATCH"; exit
}
trap finish EXIT

# Run psi4
export OMP_NUM_THREADS=4
srun -n1 $PSI4 -s "$PSI4_SCRATCH" -n25
