import os
from PIL import Image

def convert_to_jpg(folder_path):
    # 支持的文件扩展名（不区分大小写）
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for filename in os.listdir(folder_path):
        # 获取文件扩展名并转为小写
        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        
        if ext_lower not in valid_extensions:
            continue  # 跳过不支持的文件

        file_path = os.path.join(folder_path, filename)
        new_filename = f"{name}.jpg"
        new_file_path = os.path.join(folder_path, new_filename)

        try:
            # 打开图片并处理透明背景
            with Image.open(file_path) as img:
                # 转换为RGB模式（解决RGBA/P模式的兼容性问题）
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 保存为JPG（质量设为85%以平衡大小和画质）
                img.save(new_file_path, 'JPEG', quality=85)
                
                # 删除原始非JPG文件（保留原JPG文件）
                if ext_lower != '.jpg':
                    os.remove(file_path)
                    print(f"转换成功: {filename} -> {new_filename}")
                else:
                    print(f"跳过JPG文件: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 失败: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python switch2.py <文件夹路径>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print("错误: 提供的路径不存在或不是文件夹")
        sys.exit(1)
    
    convert_to_jpg(folder_path)