from c2net.context import prepare, upload_output
import os
import shutil

import subprocess

# 创建命令序列
commands = """
git clone https://github.com/youfeng1024/MASTER.git
cd MASTER
wget -O opensource.zip -c https://dlink.host/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8wMWI3Nzg3OWYwZDUxNzZlL0VWRzhFQWZfNVpoRnZmclZjT3ZWSUxJQlZhbmR0VjNJelJseTdobWhPckxTQUE_ZT0yVFNlZmI.zip
unzip opensource.zip -d data/
python main.py | tee output.txt
"""

# 启动一个 Bash 会话
process = subprocess.Popen(
    ["bash", "--init-file", "~/.bashrc"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1  # 行缓冲
)

# 向 Bash 会话写入命令并关闭输入流
process.stdin.write(commands)
process.stdin.close()

# 实时读取输出并打印到控制台
for line in process.stdout:
    print(line, end="")  # 实时输出到控制台

# 等待进程结束
process.wait()

c2net_context = prepare()
output_path = c2net_context.output_path
# 获取当前工作目录
current_directory = os.getcwd()

# 复制整个目录树
shutil.copytree(current_directory, output_path, dirs_exist_ok=True)
print(f"Copied {current_directory} to {output_path}")

upload_output()