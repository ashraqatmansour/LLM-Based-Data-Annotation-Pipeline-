import json 
import random 
import argparse



def extract_to_tsv(input_file, output_file,sample_size=50):
    with open(input_file, 'r') as f: 
        data = json.load(f)


    # where data is the parent and we want to extract the subheading titled "children"
    data = data['data']['children']

    print(f"total data is {len(data)}")
    
    if sample_size > len(data): 
        sample_size = len(data)
    
    print(f"sample size is {sample_size}")

    sampled_data = random.sample(data, sample_size)

    with open(output_file, 'w') as f: 
        
        f.write("Name\tTitle\tCoding\n")
        for element in sampled_data: 
            title = element['data']['title']
            name = element['data']['author_fullname']

            f.write(f"{name}\t{title}\t\n")



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Extract random samples from a JSON file to TSV file")
    parser.add_argument("-o", "--output", required=True, help="path to the desired output file")
    parser.add_argument("-i", "--input", required=True, help="Path to the desired input file")
    parser.add_argument("-s", "--sample_size", type=int, default=50, help="Number of random samples")
    args = parser.parse_args()

    extract_to_tsv(args.input, args.output, args.sample_size)