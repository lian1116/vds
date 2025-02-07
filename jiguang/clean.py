import os

def delete_files_in_subfolders(root_dir):
    """
    删除当前文件夹内所有子文件夹中的所有文件，但保留 clean.py 文件
    :param root_dir: 当前文件夹路径
    """
    # 遍历当前文件夹中的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 遍历子文件夹中的所有文件
        for filename in filenames:
            # 跳过 clean.py 文件
            if filename == "clean.py":
                print(f"跳过 clean.py 文件：{filename}")
                continue

            file_path = os.path.join(dirpath, filename)
            try:
                # 删除文件
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            except Exception as e:
                print(f"删除文件失败：{file_path}，错误：{e}")

# 使用示例
if __name__ == "__main__":
    # 当前文件夹路径
    current_dir = os.getcwd()  # 获取当前工作目录
    print(f"当前文件夹：{current_dir}")

    # 删除当前文件夹内所有子文件夹中的所有文件
    delete_files_in_subfolders(current_dir)