# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # List of external packages to install
# packages = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "geopy"
# ]

# if __name__ == "__main__":
#     for package in packages:
#         install(package)
#     print("All packages installed successfully.")

import subprocess
import sys
import importlib.util

def is_installed(package_name):
    # Try to find the module and return whether it's found
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def install(package):
    if is_installed(package):
        pass
        #print(f"{package} is already installed.")
    else:
        # If not installed, install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed.")

# Mapping of package names to their standard import names if they differ
package_to_import_name = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "geopy": "geopy",
    "math": "math",
    "folium": "folium",
    "scikit-learn": "scikit-learn",
    "simpy": "simpy",
    "tqdm": "tqdm",
    "imageio": "imageio",
    "nbformat": "nbformat",
    "geopandas": "geopandas",
    "imageio[pyav]": "imageio[pyav]",
    "imageio[opencv]": "imageio[opencv]",
}

if __name__ == "__main__":
    for package, import_name in package_to_import_name.items():
        install(import_name)
    print("All packages are up to date.")
