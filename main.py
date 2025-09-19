import subprocess

# Run this file to execute everything. 

if __name__ == "__main__":

    # List of module directories to run
    modules = ["about_you", "drinking", "nicotine", "drugs", "location", "weed"]

    for module in modules:
        print(f"Running module: {module}")
        subprocess.run(["python3", f"sections/{module}/{module}.py"])
        print(f"Finished module: {module}\n")

