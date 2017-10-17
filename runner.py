import os

files = [f for f in os.listdir(".") if "auto-" in f]

for file in files:
    cmd = "python " + file + "&"
    # os.system(cmd)
    print cmd