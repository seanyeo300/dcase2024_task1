# This script is designed to run a specified Python script multiple times with different checkpoint IDs and experiment names.
    
import subprocess
import time
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
    ### DELAY CONFIGURATION ###
    # DELAY_SECONDS = 12000  # Delay for 5 minutes (set to 0 to skip)
    DELAY_SECONDS = 0 # Delay by 1 run (set to 0 to skip)
    print(f"[INFO] Delaying execution for {DELAY_SECONDS} seconds...")
    time.sleep(DELAY_SECONDS)
    # Define the script to run
    # script_name = 'run_training_KD_gpu_h5_tv1_ensemble_TA.py'
    # script_name = 'run_training_KD_gpu_h5_tv1.py'
    # script_name = 'run_training_DynMN_h5_KD_tv1.py'
    # script_name = 'run_training_KD_logit_stand_h5_tv1.py'
    # script_name = 'run_training_KD_logit_stand_h5_tv2.py'
    # script_name = 'run_training_KD_logit_stand_h5_tv3.py'
    # script_name = 'run_training_KD_logit_stand_h5_tv4.py'
    script_name = 'run_training_KD_gated_h5_tv1.py'
    # script_name = 'run_training_KD_wsum_h5_tv1.py'
    # script_name = 'run_training_KD_teacher_logit_stand_h5_tv1.py'
    # script_name = 'run_training_DynMN_h5_KD_logit_stand_tv1.py'
    
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ['--gpu','[0]',"--subset", "5", "--dir_prob", "0.6", "--mixstyle_p", "0.4","--kd_lambda","0.02","--temperature","3","--logit_stand"]# "--batch_size", "48","--pretrained","--model_name", "dymn10_as"] # this is for the KD process, does not apply to teachers!!! "--logit_stand"
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [
        # ("fskag87u", "NTU_KD_Var1-T_DSIT-S_FMS_DIR_sub5_fixh5"),                               #DSIT
        # ("leguwmeg", "NTU_KD_Var1-T_SIT-S_FMS_DIR_sub5_fixh5"),                                #SIT FMS DIR
        # ("dbl1yun4", "NTU_KD_Var1-T_SIT-S_FMS_sub5_fixh5"),                                    #SIT FMS
        # ("lm7o54or", "NTU_KD_Var1-T_SeqFT-S_FMS_DIR_sub5_fixh5"),                              #SeqFT
        # ("ke771aaz", "NTU_KD_Var1-T_FTtau-S_FMS_DIR_sub10_fixh5"),                             #FTtau FMS DIR
        # ("y7frm0sm", "NTU_KD_Var1-T_FTtau-S_FMS_sub5_fixh5"),                                  #FTtau FMS
        # ("eqov5ca2", "NTU_KD_Var1-T_FTtau-S_FMS_DIR_Mixup_sub5_fixh5"),                        #FTtau FMS DIR MIXUP
        # (None, "NTU_KD_tv1b-T_32BCBL-S_FMS_DIR_sub5_fixh5")                    #tv1b
                                                                                                 #tv3b== 6 SIT Ensemble, same augs
        # (None, "NTU_KD_Dy10TA1-T_32BCBL-S_FMS_DIR_sub5_fixh5")                                 #TA1
        # (None, "NTU_KD_tv1b-T_DyMN20-TA_FMS_DIR_sub5_fixh5")                                   # Dymn20 tv1b
        # (None, "NTU_KD_DyTA1-T_32BCBL-S_FMS_DIR_sub5_fixh5")                                   #TA1->BCBL distillation 
        # (None, "NTU_KD_Dy20TA1-TA_Dy10TA1-TA_FMS_DIR_sub5_fixh5")                              # tv1->Dymn20->Dymn10 T=2, lmbda=0.02
        # (None, "NTU_KD_Dy20TA1-TA_Dy10TA1-TA_FMS_DIR_T=4_lmbda=0.05_sub5_fixh5")               # tv1->Dymn20->Dymn10 logit stand, T=3, lmbda=0.05
        # (None, "NTU_KD_EnDy20TA1-T_32BCBL-S_FMS_DIR_sub5_fixh5")                               # DyMN20 TA1 Ensemble -> BCBL
        # (None, "NTU_KD_3SIT3BCBL-T_32BCBL-S_FMS_DIR_stand_T=2_lmbda=0.06_sub5_fixh5")          # DCASE SIT BCBL logit stand, T=3, lmbda=0.05
        # (None, "NTU_KD_3SIT3BCBL_mixaug-T_32BCBL-S_FMS_DIR_stand_T=3_lmbda=0.03_sub5_fixh5")   # DCACSE SIT BCBL NAIVE TEMP
        # (None, "NTU_KD_3SIT3BCBL-T_32BCBL-S_FMS_DIR_T=3_lmbda=0.05_sub5_fixh5")                # DCASE SIT BCBL T=3, lmbda = 0.05
        # (None, "NTU_KD_3PaSST3BCBL-T_32BCBL-S_FMS_DIR_stand_T=3_lmbda=0.05_sub5_fixh5")        # DCASE PaSST BCBL logit stand, T=3, lmbda=0.05
        # (None, "NTU_KD_3PaSST3BCBL-T_32BCBL-S_FMS_DIR_sub5_fixh5")                             # DCASE PaSST BCBL same augs
        # (None, "NTU_KD_3PaSST3BCBL_mixaug-T_32BCBL-S_FMS_DIR_sub5_ali_fixh5")                  # DCASE PaSST BCBL mix augs
        # (None, "NTU_KD_3PaSST3BCBL_mixaug-T_32BCBL-S_FMS_DIR_nomixup_sub5_fixh5")                # NTU PaSST BCBL mix augs no BCBL mixup
        # (None, "NTU_KD_2PaSST2SIT2BCBL-T_32BCBL-S_FMS_DIR_sub5_ali_fixh5")                        # DCASE PaSST BCBL mix augs
        # (None, "NTU_KD_6SIT6CPR-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                 # DCASE SIT CPR T=2, lmbda=0.02
        # (None, "NTU_KD_6PASST6CPR-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_retest_fixh5")                   # DCASE PaSST CPR T=2, lmbda=0.02
        # (None, "NTU_KD_6SIT6BCBL-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE SIT BCBL T=2, lmbda=0.02 mixed augs
        # (None, "NTU_KD_6SIT6BCBL_mixaug-T_32BCBL-S_FMS_DIR_wsum_T=3_lmbda=0.04_sub5_fixh5")                   # DCASE SIT BCBL T=3, lmbda=0.02 mixed augs
        (None, "NTU_KD_6SIT6BCBL_mixaug-T_32BCBL-S_FMS_DIR_max_vote_T=3_lmbda=0.02_sub5_fixh5")                   # DCASE SIT BCBL T=3, lmbda=0.02 mixed augs
        # (None, "NTU_KD_6SIT6BCBL_sameaugs-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE SIT BCBL T=2, lmbda=0.02 same augs
        # (None, "NTU_KD_6PaSST6BCBL_mixaugs-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE PaSST BCBL T=2, lmbda=0.02 mix augs
        # (None, "NTU_KD_6PaSST6BCBL_sameaugs-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE PaSST BCBL T=2, lmbda=0.02 same augs
        # (None, "NTU_KD_6BCBL-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE 6 BCBL T=2, lmbda=0.02 Same Augs
        # (None, "NTU_KD_12SIT-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE SIT T=2, lmbda=0.02
        # (None, "NTU_KD_12PaSST-T_32BCBL-S_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE PaSST T=2, lmbda=0.02
        # (None, "NTU_KD_6PaSST_6CPR-T_32BCBL-S_sameaugs_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE PaSST T=2, lmbda=0.02
        # (None, "NTU_KD_6SIT_6CPR-T_32BCBL-S_sameaugs_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5_retest")                   # DCASE PaSST T=2, lmbda=0.02
        # (None, "NTU_KD_3SIT3CPR-T_32BCBL-S_mixaugs_FMS_DIR_T=2_lmbda=0.02_sub5_fixh5")                   # DCASE PaSST T=2, lmbda=0.02
        # (None, "NTU_KD_6SIT6CPR-T_32BCBL-S_mixaugs_FMS_DIR_stand_T=3_lmbda=0.01_sub5_fixh5")                   # DCASE PaSST T=2, lmbda=0.02
        # (None, "NTU_KD_tv1b-T_32BCBL-S_FMS_DIR_stand_lmda=0.1_sub5_fixh5") #tv1 logit stand
        # (None, "NTU_KD_single_SIT-T_32BCBL-S_FMS_DIR_sub5_fixh5")
        # (None, "NTU_KD_3SIT3BCBL-T_32BCBL-S_FMS_DIR_max_vote_T=3_lmbda=0.01_sub5_fixh5")          # DCASE PaSST BCBL gated, T=3, lmbda=0.02
        # (None, "NTU_KD_3SIT3BCBL-T_32BCBL-S_FMS_DIR_wsum_T=3_lmbda=0.01_sub5_fixh5")          # DCASE SIT BCBL gated, T=3, lmbda=0.02
    ]
    
    # Number of times to repeat each experiment
    num_repeats = 1

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    