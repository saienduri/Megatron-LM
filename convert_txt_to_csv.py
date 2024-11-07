import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, default='Megatron-LM/logs/summary_0.txt') 
parser.add_argument('output', type=str, default='output.csv') 

args = parser.parse_args()

# Input and output file paths
input_file = args.input
output_file = args.output

headers = ["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg", "Self CUDA", "Self CUDA %", "CUDA total", "CUDA time avg", "# of Calls"]

# Initialize lists to store data
data = []

starts = [0]
# Read the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

header_line = lines[1]
for header in headers:
    index = header_line.index(header)
    starts.append(index + len(header))
    header_line = header_line[:index] + '*'*len(header) + header_line[index + len(header):]
    

data = [headers]
time_headers = [2, 4, 5, 6, 8, 9]
# Process each line of the input
for line in lines[3:-4]:
    fields = []
    for i in range(len(headers)):
        field = line[starts[i]:starts[i+1]]
        field = field.strip()
        if i in time_headers:
            if 'us' in field:
                field = field[:-2]
                field = float(field)
            elif 'ms' in field:
                field = field[:-2]
                field = float(field)*1000
            else:
                try:
                    field = field[:-1]
                    field = float(field)*1000*1000
                except:
                    print(field)
        fields.append(field)
    data.append(fields)

# Write the extracted data to CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)  # Write data rows

print(f"CSV file '{output_file}' has been generated successfully.")
