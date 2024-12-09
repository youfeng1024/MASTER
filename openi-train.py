from c2net.context import prepare, upload_output
import os
import shutil

import subprocess

# 创建命令序列
commands = """
git clone https://github.com/youfeng1024/MASTER.git
cd MASTER
wget -O opensource.zip -c https://dlink.host/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8wMWI3Nzg3OWYwZDUxNzZlL0VWRzhFQWZfNVpoRnZmclZjT3ZWSUxJQlZhbmR0VjNJelJseTdobWhPckxTQUE_ZT0yVFNlZmI.zip 
unzip -n opensource.zip -d data/
python -u main.py | tee output.log
"""

# 启动一个 Bash 会话
process = subprocess.Popen(
    ["bash", "--init-file", "~/.bashrc"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,  # 行缓冲
    universal_newlines=True  # 使用文本模式
)

# 向 Bash 会话写入命令并关闭输入流
process.stdin.write(commands)
process.stdin.close()

# 实时读取输出
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# 读取错误输出（如果有）
if process.stderr != None:
    stderr_output = process.stderr.read()
    if stderr_output:
        print("Error Output:\n", stderr_output)

# 确保进程结束
process.wait()

c2net_context = prepare()
output_path = c2net_context.output_path
# 获取当前工作目录
current_directory = os.getcwd()

# 复制整个目录树
shutil.copytree(current_directory, output_path, dirs_exist_ok=True)
print(f"Copied {current_directory} to {output_path}")

upload_output()