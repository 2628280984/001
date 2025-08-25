import os

# 输入和输出的根文件夹路径
input_folder = 'EgoGesture/rgb'  # 替换为你的文件夹路径
output_folder = 'EgoGesture/depth'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 递归遍历文件夹中的所有文件
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.txt'):  # 只处理 .txt 文件
            input_file = os.path.join(root, file)
            # 构建对应的输出文件路径
            relative_path = os.path.relpath(input_file, input_folder)
            output_file = os.path.join(output_folder, relative_path)

            # 确保输出文件夹存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # 读取文件内容
            with open(input_file, 'r') as f_in:
                lines = f_in.readlines()

            # 修改每一行，将 'Color/rgb' 替换为 'Depth/depth'
            modified_lines = []
            for line in lines:
                parts = line.strip().split(' ', 1)  # 只分割一次，得到路径和其余部分
                if len(parts) > 1:
                    parts[0] = parts[0].replace('Color/rgb', 'Depth/depth')  # 替换第一列中的路径
                    modified_line = ' '.join(parts)  # 重新组合行
                else:
                    modified_line = parts[0]  # 如果只有一列，不做修改
                modified_lines.append(modified_line)

            # 将修改后的内容写入新的文件
            with open(output_file, 'w') as f_out:
                f_out.writelines('\n'.join(modified_lines) + '\n')  # 确保每行以换行符结束

print("所有文件已经处理完成，路径已替换并保存在输出文件夹中。")
