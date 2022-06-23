from glob import glob

dataset = "hate"
path = f"/home/yila22/prj/attention/test_outputs/{dataset}/average*/*"
import os
print(dataset)
for sd in glob(path):
    import json
    print(sd)
    with open(os.path.join(sd,'evaluate.json')) as f:
        d = json.load(
            f
        )
        print(d['macro avg/f1-score'])