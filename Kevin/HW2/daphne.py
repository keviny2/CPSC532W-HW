import json
import subprocess

# TODO: change cwd path to reflect moving daphne folder
def daphne(args, cwd='../../../daphne'):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd, shell=True)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)

