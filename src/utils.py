import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content