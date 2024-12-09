from c2net.context import prepare, upload_output
import os
import shutil

import os

# Clone the repository
os.system("git clone https://github.com/youfeng1024/MASTER.git")

# Change working directory to the cloned repository
os.chdir("MASTER")

# Download the dataset
os.system("curl -u youfeng1024@gmail.com:ycziyobue6ir3l2e -O https://app.koofr.net/dav/OneDrive/dataset/opensource.zip")

# Unzip the dataset
os.system("unzip opensource.zip -d data/opensource/")

# Run the main script and save output
os.system("python main.py | tee output.txt")

c2net_context = prepare()
output_path = c2net_context.output_path
# 获取当前工作目录
current_directory = os.getcwd()

# 复制整个目录树
shutil.copytree(current_directory, output_path, dirs_exist_ok=True)
print(f"Copied {current_directory} to {output_path}")

upload_output()