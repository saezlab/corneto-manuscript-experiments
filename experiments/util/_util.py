import importlib.util
import subprocess
import os

def get_git_commit_hash(package_name):
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.origin:
        return f"Package {package_name} not found"

    package_path = os.path.dirname(spec.origin)
    git_dir = os.path.abspath(os.path.join(package_path, ".."))

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=git_dir, text=True
        ).strip()
        return f"Current commit hash for {package_name}: {commit_hash}"
    except subprocess.CalledProcessError:
        return f"Not a git repository: {git_dir}"
