from pathlib import Path
import os

def get_project_root():

    i = 3 #Since in my project structure are max at 3 level
    parent_path = Path()
    while i > 0:  

        if Path.exists(parent_path / 'data'):
            return parent_path
        else:
            parent_path = parent_path.resolve().parent
        
        i = i - 1

    return parent_path

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))