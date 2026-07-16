import os
import subprocess
from pathlib import Path

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_files():
    status = run_cmd("git status -s")
    files = []
    for line in status.split('\n'):
        if line:
            status_chars = line[:2]
            filepath = line[3:].strip()
            if 'D' in status_chars:
                files.append((filepath, "delete"))
                continue
            
            if filepath.endswith('/'):
                for root, _, f_names in os.walk(filepath):
                    for f in f_names:
                        files.append((os.path.join(root, f), "add"))
            else:
                files.append((filepath, "add"))
    return files

def commit_file(filepath, action="add"):
    path = Path(filepath)
    name = path.stem.replace('_', ' ')
    
    category = "file"
    if "middleware" in filepath:
        category = "middleware"
    elif "services" in filepath:
        category = "service"
    elif "routes" in filepath:
        category = "route file"
    elif "utils" in filepath:
        category = "util"
    elif "schemas" in filepath:
        category = "schema"
    elif "workers" in filepath:
        category = "worker"
    elif "tests" in filepath:
        category = "test"
    elif "migrations" in filepath:
        category = "migration"
    elif "models" in filepath:
        category = "model"
    elif "clients" in filepath:
        category = "client"
    elif "ai" in filepath:
        category = "ai module"
    elif "pipelines" in filepath:
        category = "pipeline module"
    elif "training" in filepath:
        category = "training module"
    elif "config" in filepath:
        category = "config"
    elif filepath in ["README.md", ".gitignore", "ci.yml", "Jenkinsfile", "main.py", "Dockerfile", "pyproject.toml", "requirements.txt", "requirements-dev.txt", ".env.example", "alembic.ini", "docker-compose.yml"] or filepath.endswith(".sh"):
        category = "config file"
    
    if filepath.endswith(".pyc"):
        return
        
    verb = "remove" if action == "delete" else "add"
    msg = f"feat: {verb} {category} to the vision ai: {name}"
    
    print(f"Committing {filepath} with message: {msg}")
    
    if action == "delete":
        run_cmd(f"git rm {filepath}")
    else:
        run_cmd(f"git add {filepath}")
        
    run_cmd(f'git commit -m "{msg}"')

files_info = get_files()

# Order: middlewares, services, routes, utils, others
order_categories = ["middleware", "services", "routes", "utils"]

for cat in order_categories:
    cat_files = [(f, a) for f, a in files_info if cat in f]
    for f, a in cat_files:
        commit_file(f, a)

# others
other_files = [(f, a) for f, a in files_info if not any(cat in f for cat in order_categories)]
for f, a in other_files:
    commit_file(f, a)
