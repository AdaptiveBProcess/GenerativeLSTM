import subprocess
import os

def call_simod(log_name):
    print('Executing Simod with {} log.'.format(log_name))
    os.chdir("Simod-Coral-Version")
    subprocess.run(["python", "simod_optimizer.py", "-f", log_name, "-m", "sm3"])
    print('Execution ended.')