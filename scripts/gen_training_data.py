#microsoft/cats_vs_dogs
for training_type in ["noise","scale","scale_vae"]:
    command=f"sbatch -J scale --err=slurm_chip/trainingv1/{training_type}_catdog.err --out=slurm_chip/trainingv1/{training_type}_catdog.out "
    command+=f" --constraint=L40s runpygpu_chip_l40.sh "
    #command+=f" runpygpu_chip.sh "
    command+=f"main.py --save_dir {training_type}_catdog --repo_id jlbaker361/{training_type}_catdog --batch_size 4 --val_interval 10 --training_type {training_type} --limit -1 --epochs 100 --gradient_accumulation_steps 8 "
    command+=f" --hf_path_data microsoft/cats_vs_dogs "
    print(command)