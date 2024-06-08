import os
def load_confidence(confidence_folder):
    """
    Load confidence values from all files in the specified folder.
    
    Args:
    confidence_folder (str): Path to the folder containing confidence files.

    Returns:
    list: List of confidence values extracted from the files.
    """
    confidence_files = sorted(os.listdir(confidence_folder))
    confidence_list = []
    for confidence_file in confidence_files:
        with open(os.path.join(confidence_folder,confidence_file), 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            print(parts[-1])
            confidence_list.append(parts[-1])

    return confidence_list
         
def modify_file(input_file, output_file, confidence_list):
    """
    Modify the contents of the input file based on certain rules and save to the output file.

    Args:
    input_file (str): Path to the input file.
    output_file (str): Path to the output file.
    confidence_list (list): List of confidence values to replace specific columns.
    """

    with open(input_file, 'r') as file:
        lines = file.readlines()
    num = 0
    modified_lines = []
    for line in lines:
        parts = line.strip().split(',')
        # # print(parts)
        if len(parts) > 1:
            parts[1] = str(int(parts[1]) + 1)  # Increment the second value by 1
        
        # Round and format specific columns (columns 2, 3, 4, 5 in zero-based index)
        parts[2] = f"{abs(round(float(parts[2]), 2)):.2f}"
        parts[3] = f"{abs(round(float(parts[3]), 2)):.2f}"
        parts[4] = f"{abs(round(float(parts[4]), 2)):.2f}"
        parts[5] = f"{abs(round(float(parts[5]), 2)):.2f}"
        parts[-4] = confidence_list[num] # Replace the specific column with a confidence value
        # parts[-4] = f'0.9'
        num += 1
        


        modified_lines.append(','.join(parts))

    with open(output_file, 'w') as file:
        for line in modified_lines:
            file.write(line + '\n')

# Iterate over all files in the specified subfolder and modify them
subfolder_path = "MOT15/MULTI_MATCH_RESULT"
confidence_path = '../detect_results'

for subfolder_name in os.listdir(subfolder_path):
    
    confidence_folder = subfolder_name.split('.')[0]
    print(confidence_folder)
    input_file = f"MOT15/MULTI_MATCH_RESULT/{subfolder_name}"
    output_folder = os.path.join(f'postprocess',f'{subfolder_path}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder,f"{subfolder_name}")
    
    confidence_list = load_confidence(os.path.join(confidence_path,confidence_folder))
    modify_file(input_file, output_file, confidence_list)
    print(f"Modified file saved as {output_file}")