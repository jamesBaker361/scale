for training_type in ["noise","scale","scale_vae"]:
    command=f"sbatch --err=slurm_chip/trainingv1/{training_type}.err --out=slurm_chip/trainingv1/{training_type}.out "
    command+=f" runpygpu_chip.sh main.py --save_dir {training_type} --name jlbaker361/{training_type} "
    print(command)