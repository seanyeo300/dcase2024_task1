# multi_run.py
import subprocess

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

def run_multiple_scripts(scripts_with_args):
    try:
        for script_name, args in scripts_with_args:
            # Convert all arguments to strings
            args = [str(arg) for arg in args]
            subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names and arguments
    scripts_to_run = [
        ('run_training_FMS_DIR2.py', ['--ckpt_id', None, "--experiment_name","tBCBL_sub5_441K_32_channel_STtau_DIR_only_nh5"]),
        ('run_training_FMS_DIR2.py', ['--ckpt_id', None, "--experiment_name", "tBCBL_sub5_441K_32_channel_STtau_DIR_only_nh5"]),
        ('run_training_FMS_DIR2.py', ['--ckpt_id', None, "--experiment_name", "tBCBL_sub5_441K_32_channel_STtau_DIR_only_nh5"])
    ]
    run_multiple_scripts(scripts_to_run)    