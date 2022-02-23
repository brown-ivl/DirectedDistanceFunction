import os

output = os.popen("ssh thouchen@ssh.ccv.brown.edu ls /gpfs/data/ssrinath/neural-odf/output").read().split("\n")
print("+++"*20)
print(output)
