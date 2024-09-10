# Loop to run 5 times, you can adjust this as needed

#sbatch slurm_job_databrick_8_10.sh
#sleep 2100
#sbatch slurm_job_databrick_8_5.sh
#sleep 2400
#sbatch slurm_job_databrick_8_2.sh
#sleep 2700

while true
do
  # command_to_run
  # Sleep for 30 minutes
  # sbatch slurm_job_full_del.sh
  sbatch slurm_job_databrick_del.sh
  #sleep 2700
  sleep 10800
done
