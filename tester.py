import os
import yaml
import subprocess

def update_yaml(file_path, max_attack_ratio, label_attack_ratio):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['max_attack_ratio'] = max_attack_ratio
    config['label_attack_ratio'] = label_attack_ratio

    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

yaml_file = r'C:\Users\kunal\federated-ml\conf\rf.yaml'  # Use raw string for Windows path

output_dir = 'output'  # Directory where output files will be saved

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for it_client_attack_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for it_label_attack_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        max_attack_ratio = it_client_attack_ratio 
        label_attack_ratio = it_label_attack_ratio 
        
        # Generate a unique filename based on ratios
        output_file = os.path.join(output_dir, f"output_{it_client_attack_ratio}_{it_label_attack_ratio}.txt")

        with open(output_file, 'w') as output:
            output.write(f"client_attack_ratio: {it_client_attack_ratio}\n")
            output.write(f"label_flipping_ratio: {it_label_attack_ratio}\n")  # Add label_flipping_ratio
            
            update_yaml(yaml_file, max_attack_ratio, label_attack_ratio)
            
            result = subprocess.run([r'C:\Users\kunal\federated-ml\.venv\Scripts\python.exe', 'main.py'], capture_output=True, text=True, shell=True)
            
            output.write(result.stdout + '\n')
            
            # Save the terminal output to the file
            output.write(f"Terminal Output:\n{result.stderr}\n")
