import os
import shutil


def copy_files(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)

        # 如果是文件而不是文件夹，则进行复制
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"已复制: {source_file} -> {destination_file}")


# 设置源文件夹和目标文件夹路径
source_folder = "news_articles"
destination_folder = "data3"

# 执行复制操作
copy_files(source_folder, destination_folder)
print("文件复制完成！")