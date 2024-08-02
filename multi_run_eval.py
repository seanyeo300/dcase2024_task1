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
        ('get_logits.py', ['--evaluate', '--ckpt_id', '1o0lgb4u']),
        ('get_logits.py', ['--evaluate', '--ckpt_id', 'dhrckv13']),
        ('get_logits.py', ['--evaluate', '--ckpt_id', 'l521lxpv'])
        # ('get_logits.py', ['--evaluate', '--ckpt_id', '4zesr7bt'])
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'fm67hz2c'])
    ]
    run_multiple_scripts(scripts_to_run)

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