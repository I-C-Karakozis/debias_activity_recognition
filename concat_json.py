import argparse
import json

# Sample execution: 
# python concat_json.py data/test.json data/dev.json data/concat_test.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two json files into a single output_json.") 
    parser.add_argument("input_json1") 
    parser.add_argument("input_json2")
    parser.add_argument("output_json")
    args = parser.parse_args()        

    input1 = json.load(open(args.input_json1))
    input2 = json.load(open(args.input_json2))
    output = {key: value for (key, value) in (input1.items() + input2.items())}
    with open(args.output_json, "w") as f:     
        json.dump(output, f)
