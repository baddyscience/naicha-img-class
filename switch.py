import os

def rename_photos(folder_path):
    # 支持的图片扩展名列表（小写）
    supported_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    # 收集符合条件的图片文件
    files = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_ext:
                files.append(filename)
    
    # 按文件名排序以避免随机顺序
    files.sort()
    
    # 第一步：将文件重命名为临时文件名，避免覆盖冲突
    temp_files = []
    for filename in files:
        src = os.path.join(folder_path, filename)
        temp_name = f"temp_{filename}"
        dst = os.path.join(folder_path, temp_name)
        try:
            os.rename(src, dst)
            temp_files.append(temp_name)
        except Exception as e:
            print(f"错误：无法重命名 {filename} 为临时文件。原因：{e}")
            return  # 遇到错误时终止
    
    # 第二步：将临时文件按顺序重命名为 yht1, yht2...
    counter = 1
    for temp_name in temp_files:
        ext = os.path.splitext(temp_name)[1].lower()
        new_name = f"mx{counter}{ext}"
        src = os.path.join(folder_path, temp_name)
        dst = os.path.join(folder_path, new_name)
        try:
            os.rename(src, dst)
            print(f"已重命名：{temp_name} -> {new_name}")
            counter += 1
        except Exception as e:
            print(f"错误：无法重命名 {temp_name} 为 {new_name}。原因：{e}")

if __name__ == "__main__":
    target_folder = input("请输入目标文件夹路径：")
    if os.path.isdir(target_folder):
        rename_photos(target_folder)
        print("重命名完成！")
    else:
        print("错误：路径不存在或不是文件夹。")