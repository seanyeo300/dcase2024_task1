# multi_run.py

# def run_multiple_scripts(scripts_with_args):
#     try:
#         for script_name, args in scripts_with_args:
#             subprocess.run(['python', script_name] + args)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Example: Replace these with your actual script names and arguments
#     scripts_to_run = [
#         ('run_passt_cochl_PT_mel_h5.py', ['--lr', 1e-4]),
#         ('run_passt_cochl_PT_mel_h5.py', [ '--lr', 1e-5]),
#         ('run_passt_cochl_PT_mel_h5.py', [ '--lr', 1e-6])
#     ]
#     run_multiple_scripts(scripts_to_run)
    
#     import subprocess
import subprocess

def run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats):
    try:
        for ckpt_id, experiment_name in ckpt_experiment_pairs:
            # Update arguments with the current ckpt_id and experiment_name
            ckpt_id_arg = "None" if ckpt_id is None else ckpt_id
            args = base_args + ["--ckpt_id", ckpt_id_arg, "--experiment_name", experiment_name]
            
            # Run the script multiple times with the same arguments
            for _ in range(num_repeats):
                subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

### How to use this script###
# 1. Check the script name to ensure you are distilling from the correct teacher ensemble
# 2. Check base args for subset and augmentations
# 3. Check model variants and individual Checkpoint IDs

if __name__ == "__main__":
    # Define the script to run
    script_name = 'run_training_KD_gpu_h5_tDSIT_sub10.py' # specify which subset your DSIT model is trained on
    
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ["--subset", "5", "--dir_prob", "0.6", "--mixstyle_p", "0.4"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    
    ##########################################################################################
    ##!!Teacher is always DSIT single model, don't be confused by other multli run scripts!!##
    ##########################################################################################
    ckpt_experiment_pairs = [                                           # Students
        # ("fskag87u", "NTU_KD_DSIT-T_DSIT-S_FMS_DIR_sub5_fixh5"),     #DSIT
        # ("leguwmeg", "NTU_KD_DSIT-T_SIT-S_FMS_DIR_sub5_fixh5")       #SIT FMS DIR
        # ("dbl1yun4", "NTU_KD_DSIT-T_SIT-S_FMS_sub5_fixh5"),          #SIT FMS
        # ("lm7o54or", "NTU_KD_DSIT-T_SeqFT-S_FMS_DIR_sub5_fixh5"),    #SeqFT    
        # ("ke771aaz", "NTU_KD_DSIT-T_FTtau-S_FMS_DIR_sub10_fixh5")      #FTtau
        # (None, "NTU_KD_DSIT-T_BCBL-S_FMS_DIR_sub10_fixh5"),             #Ptau 
        ("hsxt4lnx", "NTU_KD_DSIT-T_FTtau-S_FMS_DIR_sub5_fixh5")
    ]
    
    # Number of times to repeat each experiment
    num_repeats = 1

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    
# import subprocess

# def run_multiple_scripts(scripts_with_args):
#     try:
#         for script_name, args in scripts_with_args:
#             # Convert all arguments to strings
#             args = [str(arg) for arg in args]
#             subprocess.run(['python', script_name] + args)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Example: Replace these with your actual script names and arguments
#     scripts_to_run = [
#         # ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         # ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"])
#     ]
#     run_multiple_scripts(scripts_to_run)