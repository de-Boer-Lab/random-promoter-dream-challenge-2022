import argparse
import json
import numpy as np
from scipy import stats

# For 9045 length data prediction (same as sample_submission.json)

# loc : mean of standardized expression from train data
# scale : 1
def postprocess(x):
    return stats.norm.cdf(x, loc=0.06201038511068657, scale=1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open('prediction/sample_submission.json', 'r') as inFile:
        sample = json.loads(inFile.read())
        sample_keys = set(sample.keys())
        
        assert len(sample_keys) == 9045

    with open(args.input, 'r') as inFile, open(args.output, 'w') as outFile:
        orig = json.loads(inFile.read())
        orig = {k:v for k, v in orig.items() if k in sample_keys}

        predictions = np.array(list(orig.values()))
        flattened = dict(zip(orig.keys(), postprocess(predictions)))
        print(json.dumps(flattened), file=outFile)
