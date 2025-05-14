#!/bin/bash

#SBATCH --partition=itp
#SBATCH --nodes=2
##SBATCH --ntasks=21            # Matches the number of array tasks
#SBATCH --cpus-per-task=1      # One CPU per task
#SBATCH --mem-per-cpu=6000     # 6GB per CPU
#SBATCH --job-name=jobscript
#SBATCH --output=output_logs/jobscript.out
#SBATCH --error=output_logs/jobscript.err
#SBATCH --mail-type=FAIL
#SBATCH --array=0-10:1           # Manually set the correct range


# Array of chemical potential values
chemical_potential_vals=(-5.9 -5.3 -4.6 -4.5 -4.4 -4.3 -4.2 -4.1 -4.0 -3.9)

# chemical_potential_vals=(-5.2)
mkdir -p output_logs

val="${chemical_potential_vals[$SLURM_ARRAY_TASK_ID]}"
echo "Running with chemical potential: $val"
python3.12 -u main.py "$val" > "output_logs/output_$val.log" 2>&1 

wait
echo "All jobs finished."