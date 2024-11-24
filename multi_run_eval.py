# multi_run.py
import subprocess

def run_multiple_scripts(scripts_with_args):
    try:
        for script_name, args in scripts_with_args:
            subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names and arguments
    scripts_to_run = [
        ('get_logits.py', ['--evaluate', '--ckpt_id', 'mtkxd1f9', '--sample_rate', '44100']),
        # ('get_logits.py', ['--evaluate', '--ckpt_id', "i9r5u5bz", '--sample_rate', '44100']),
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'c7urqd64', '--sample_rate', '44100']),
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'phbu0vl2','--subset','5',"--base_channels", "96"]),
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'hx1xuegl','--subset','5',"--base_channels", "96"])
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','4kvh9p1r','--gpu','[1]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"]),
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','0lx0k8pp','--gpu','[0]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"] ),
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','igl8n28c','--gpu','[1]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"]),
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','kiykj1zl','--gpu','[0]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"]),
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','yerjwwdq','--gpu','[1]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"] ),
        # ('run_training_DynMN_h5_PL.py', ['--evaluate','--ckpt_id','h3ez8bgq','--gpu','[0]',"--subset", "5", "--batch_size", "48","--model_name", "dymn20_as","--model_width","2"])
    ]
    run_multiple_scripts(scripts_to_run)


# tBCBL_sub5_nh5
['7og6lmpb', '18llaju7', 'gc41eep8', 'r0ina4uc', '32jp9l7h', 'nwg60o2k']
#BCBL_sub5 FMS+DIR, FMS, DIR
["7og6lmpb", "kybgcsn2", "7ltm2p1l"]
# tBBL_sub10_nh5
['qi2qx93m','gxw02np0', 'w1cm7obb','6gy1hz5n','0lumb7qw','4y5iqp12']


#Dy20TA1
'ncalnvch','njms2d1h','taoi4zzh','19vmadkn','p6lmh9x0','8uu9o45m'

#Dy20TA2
'lo8xn4i3','vt3dl9hx','6oz7k7j9','mvitmq7r','o70onxc7','14bjpp51'

#Dy20TA3
'8dzf1v5z','0lx0k8pp','igl8n28c','kiykj1zl','yerjwwdq','h3ez8bgq'

#Dy10TA1
'm1gyrwjn'
'o369g0wu'
'iy6mwlo5'
'jbtddev3'
'ec0bjwrz'
'i5ubtkhn'

#Dy10TA2
'r46xbxe4'
'rv0v6e4m'
'f5um2bip'
'7podxlyv'
'29njyans'
'hv5kjdv4'

#Dy10TA3
'sqxhajdv'
'rtafwsli'
'73xsgrfs'
'xn1jvtvt'
'p4ms9el7'
'hxj6baec'








###32-BCBL@44.1K models###
#50%
["3xlli7dq", "wpu6shc2", "8cbb6x4v"]
#25%
["b9ooz0ks", "dksmk72n", "29rio0lo"]    #done  
#10%
['pw3jremw', 'baeix291', '7qouvmdh']    #done
#5%
["nmwun6cs", "1e5ld4y6", "gs5hm18o"]

####24-BCBL@44.1K models####
['1o0lgb4u','dhrckv13','l521lxpv']


##KD-Ensemble##
#100%
['mqzabiyn'] # done
#50%
['fu9y8b7i'] # ['p22qh0f4']
#25%
['1ctisl36']
#10%
['o70ikpgn']
# 5%
['75h28rxp']

# 100% 32-BCBL 44.1K
['dj5zbrid']

#100% KD-Ensemble models
['mqzabiyn','fgw3yxt3','wyqwic3z']
# 100% TFS models
['100i1zrj','4zesr7bt','fm67hz2c']
# 5% KD-Ensemble models
["cxmze167","g7r7w32o","75h28rxp"]
# 5% TFS models
["6y1hmlfp", "a7zqm9g8", "o67k7os1"]