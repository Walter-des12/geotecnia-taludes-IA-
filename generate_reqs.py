import pkgutil
import subprocess
import sys
import os

# Librerías mínimas que debe buscar
TARGET_PACKAGES = [
    "streamlit", "numpy", "pandas", "matplotlib",
    "scikit-learn", "lightgbm", "shap", "plotly", "pyslope"
]

def get_installed_version(pkg):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split("Version:")[1].strip()
    except:
        return None

reqs = []

for pkg in TARGET_PACKAGES:
    version = get_installed_version(pkg)
    if version:
        reqs.append(f"{pkg}=={version}")
    else:
        reqs.append(f"{pkg}  # NOT INSTALLED")

# Forzar pyslope desde GitHub
for i, line in enumerate(reqs):
    if line.startswith("pyslope"):
        reqs[i] = "pyslope @ https://github.com/jessebonanno/pyslope/archive/refs/heads/master.zip"

# Guardar archivo
with open("requirements.txt", "w") as f:
    f.write("\n".join(reqs))

print("requirements.txt generado correctamente.")
