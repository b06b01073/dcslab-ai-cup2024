import os

import shutil
def move_folder(folder_A,folder_B):
    # folder_B = 'A/B'
    # folder_A = 'A'
    # Get a list of all files in folder B
    files = os.listdir(folder_B)
    # Loop through each file in folder B
    for file_name in files:
        # Check if the file is a .txt file
        if file_name.endswith('.txt'):
            # Create the full path to the file
            file_path = os.path.join(folder_B, file_name)
            # Move the file to folder A
            shutil.move(file_path, folder_A)
    print(f"All .txt files have been moved from {folder_B} to {folder_A}.")

def create_empty_txt_files(folder_a, folder_b):
    # Traverse through all subfolders and files in folder_a
    for subdir, _, files in os.walk(folder_a):
        # Extract relative path to maintain the same structure
        relative_path = os.path.relpath(subdir, folder_a)
        
        # Construct corresponding path in folder_b
        corresponding_b_path = os.path.join(folder_b, relative_path)
        
        # Ensure the corresponding directory exists in folder_b
        os.makedirs(corresponding_b_path, exist_ok=True)
        
        for file in files:
            # We only care about jpg files in folder_a
            if file.endswith('.jpg'):
                # Change file extension to .txt for the check
                txt_file = os.path.splitext(file)[0] + '.txt'
                
                # Path to the corresponding txt file in folder_b
                txt_file_path = os.path.join(corresponding_b_path, txt_file)
                
                # If the txt file does not exist in folder_b, create an empty one
                if not os.path.exists(txt_file_path):
                    with open(txt_file_path, 'w') as f:
                        pass  # Create an empty file
                    print(f"Created empty file: {txt_file_path}")

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python create_missing_files.py <directory>")
    #     sys.exit(1)
    folder_detect_result = "runs/detect"
    for subfolder_name in os.listdir(folder_detect_result):
        move_folder(f"{folder_detect_result}/{subfolder_name}",f"{folder_detect_result}/{subfolder_name}/labels")
        # Remove the folder_B after moving files
        shutil.rmtree(f"{folder_detect_result}/{subfolder_name}/labels")
        print(len(os.listdir(f"{folder_detect_result}/{subfolder_name}")))

    folder_AI_CUP_testdata = '../AI_CUP_testdata/images'
    
    create_empty_txt_files(folder_AI_CUP_testdata, folder_detect_result)

    
    
    



