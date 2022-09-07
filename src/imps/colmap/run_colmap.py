import os
import subprocess


def run_automoatic_reconstructor(workspace_path):
    images_path = os.path.join(workspace_path, 'images')
    process = subprocess.Popen(['colmap', 'automatic_reconstructor', f'--workspace_path {workspace_path}', f'--image_path {images_path}'], 
                                stdout=subprocess.PIPE, universal_newlines=True)

    return_code = None
    while return_code is None:
        output = process.stdout.readline()
        print(output.strip())
        return_code = process.poll()
        
    print('RETURN CODE', return_code)
    for output in process.stdout.readlines():
        print(output.strip())

    return return_code