for training_type in ["noise","scale","scale_vae"]:
    command=f"sbatch -J scale --err=slurm_chip/trainingv1/{training_type}.err --out=slurm_chip/trainingv1/{training_type}.out "
    #command+=f" --constraint=L40s runpygpu_chip_l40.sh "
    command+=f" runpygpu_chip.sh "
    command+=f"main.py --save_dir {training_type} --name jlbaker361/{training_type} --batch_size 1 "
    print(command)