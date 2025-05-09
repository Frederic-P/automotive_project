import subprocess

def install_packages(file_path='requirements.txt'):
    failed = []
    with open(file_path, 'r') as file:
        for line in file:
            package = line.strip()
            if not package or package.startswith('#'):
                continue  # Skip comments and blank lines
            print(f"Installing: {package}")
            try:
                subprocess.check_call(['pip', 'install', package])
            except subprocess.CalledProcessError:
                print(f"Failed to install: {package}")
                failed.append(package)

    if failed:
        print("\nSome packages failed to install:")
        for pkg in failed:
            print(f"- {pkg}")
    else:
        print("\nAll packages installed successfully.")

a = ''
options = ['y', 'n']
print("""This installer will try to install the necessary Python modules on your system.\n
Read the instructions in documentation/installation.md carefully before installing\n
and make sure you have your virtual environment activated before running the installation.""")
a = input("Do you want to continue with installing: (Y/N): ").lower()
while a not in options: 
    a = input("Invalid input, type 'Y' or 'N': ")
if a == 'y': 
    print('Starting installation')
    install_packages()
else:
    print('No modules installed. ')
