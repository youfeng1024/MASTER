from c2net.context import prepare, upload_output
import os
import shutil

import os

os.system("git clone https://github.com/youfeng1024/MASTER.git && cd MASTER")
os.system("curl -u youfeng1024@gmail.com:ycziyobue6ir3l2e -O https://app.koofr.net/dav/OneDrive/dataset/opensource.zip")
os.system("unzip opensource.zip -d data/opensource/")
os.system("python main.py | tee output.txt")


# 移动到输出目录
def copy_current_directory_to_output_path(output_path):
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 构建目标目录路径
    target_directory = os.path.join(output_path, os.path.basename(current_directory))

    # 确保目标路径存在，如果不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 复制整个目录树
    shutil.copytree(current_directory, target_directory, dirs_exist_ok=True)
    print(f"Copied {current_directory} to {target_directory}")

c2net_context = prepare()
output_path = c2net_context.output_path
copy_current_directory_to_output_path(output_path)

upload_output()