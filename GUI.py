import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from collections import Counter

# 禁用 GPU
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

# 参数
img_height = 200
img_width = 200
names = ['yht', 'mx', 'gm']
model_path = 'naicha-tf/checkpoint.weights.h5'

# 构建模型
def create_model():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if os.path.exists(model_path + '.index'):
        model.load_weights(model_path)
    return model

# 预测函数（启用 Dropout）
def predict_img_with_dropout(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model(img_array, training=True)
    score = tf.nn.softmax(predictions[0])
    return names[np.argmax(score)]

# 主 GUI
def launch_gui():
    def choose_file():
        filepath = filedialog.askopenfilename()
        if filepath:
            # 显示图片
            img = Image.open(filepath).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk

            # 获取用户选择的预测次数
            try:
                count = int(predict_count_var.get())
                if count <= 0:
                    raise ValueError
            except ValueError:
                tk.messagebox.showerror("错误", "请输入一个正整数")
                return

            # 加载模型
            model = create_model()

            # 执行多次预测
            results = [predict_img_with_dropout(model, filepath) for _ in range(count)]
            counter = Counter(results)

            # 清除旧图
            for widget in chart_frame.winfo_children():
                widget.destroy()

            # 绘制饼图
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie([counter[n] for n in names], labels=names, autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Percentage by category ({count})")
            ax.axis('equal')

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

    # GUI 主体
    root = tk.Tk()
    root.title("奶茶分类预测 - 多次预测分析")
    root.geometry("800x320")

    # 左侧界面
    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    tk.Label(left_frame, text="预测次数:").pack()
    predict_count_var = tk.StringVar(value="10")
    count_entry = ttk.Entry(left_frame, textvariable=predict_count_var, width=10)
    count_entry.pack(pady=5)

    btn = tk.Button(left_frame, text="选择图片并预测", command=choose_file)
    btn.pack(pady=10)

    image_label = tk.Label(left_frame)
    image_label.pack()

    # 右侧图表显示
    global chart_frame
    chart_frame = tk.Frame(root)
    chart_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()

# 启动程序
if __name__ == "__main__":
    launch_gui()
