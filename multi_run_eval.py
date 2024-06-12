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
        ('get_logits.py', ['--evaluate', '--ckpt_id', 'pw3jremw']),
        ('get_logits.py', ['--evaluate', '--ckpt_id', 'baeix291']),
        ('get_logits.py', ['--evaluate', '--ckpt_id', '7qouvmdh'])
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'pcerkmrt'])
        # ('get_logits.py', ['--evaluate', '--ckpt_id', 'duy4puj8'])
    ]
    run_multiple_scripts(scripts_to_run)
#50%
["3xlli7dq", "wpu6shc2", "8cbb6x4v"]
#25%
["b9ooz0ks", "dksmk72n", "29rio0lo"]    
#10%
['pw3jremw', 'baeix291', '7qouvmdh']
#5%
["nmwun6cs", "1e5ld4y6", "gs5hm18o"]