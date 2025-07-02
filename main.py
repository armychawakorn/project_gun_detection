import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')

def load_img():
    global original_img, img_tk
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        original_img = Image.open(file_path)
        # Resize image if too large while maintaining aspect ratio
        max_size = 600
        if original_img.width > max_size or original_img.height > max_size:
            ratio = min(max_size / original_img.width, max_size / original_img.height)
            new_size = (int(original_img.width * ratio), int(original_img.height * ratio))
            original_img = original_img.resize(new_size, Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(original_img)
        img_label.config(image=img_tk)
        status_label.config(text="รูปภาพพร้อมสำหรับการตรวจจับ")

def detect_obj():
    global original_img, img_tk_detected
    if 'original_img' in globals() and original_img:
        status_label.config(text="กำลังตรวจจับวัตถุ...")
        root.update()
        
        results = model(original_img)
        detections = results.pandas().xyxy[0]
        img_with_boxes = original_img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        for _, detect in detections.iterrows():
            x_min, y_min, x_max, y_max, confidence, class_id, class_name = detect.values
            label = f"{class_name} {confidence:.2f}"
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="#FF3366", width=2)
            text_position = (x_min, y_min - 10) if y_min > 20 else (x_min, y_min + 10)
            draw.text(text_position, label, fill="#FF3366")

        img_tk_detected = ImageTk.PhotoImage(img_with_boxes)
        img_label.config(image=img_tk_detected)
        
        if len(detections) > 0:
            status_label.config(text=f"พบวัตถุทั้งหมด {len(detections)} ชิ้น")
        else:
            status_label.config(text="ไม่พบวัตถุที่ต้องการ")
    else:
        messagebox.showinfo("Info", "กรุณาโหลดรูปภาพก่อน")

# Configure the main window
root = tk.Tk()
root.title("ระบบตรวจจับวัตถุ")
root.configure(bg='#F5F5F5')
root.geometry("700x600")

# Fonts and colors
primary_color = '#3498db'
secondary_color = '#2c3e50'
bg_color = '#F5F5F5'
accent_color = '#1abc9c'

title_font = ('Kanit', 16, 'bold')
normal_font = ('Kanit', 12)
button_font = ('Kanit', 11)

# Main container
main_frame = tk.Frame(root, bg=bg_color, padx=20, pady=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Title
header = tk.Label(main_frame, text="ระบบตรวจจับวัตถุอัจฉริยะ", font=title_font, bg=bg_color, fg=secondary_color)
header.pack(pady=(0, 15))

# Image display area
img_frame = tk.Frame(main_frame, bg=bg_color)
img_frame.pack(fill=tk.BOTH, expand=True, pady=10)

img_label = tk.Label(
    img_frame, 
    text="ไม่มีรูปภาพ โปรดกดปุ่มเพื่อโหลดรูปภาพ", 
    font=normal_font,
    bg='white', 
    height=20, 
    width=50,
    relief=tk.GROOVE, 
    borderwidth=1
)
img_label.pack(fill=tk.BOTH, expand=True)

# Status label
status_label = tk.Label(main_frame, text="พร้อมใช้งาน", font=normal_font, bg=bg_color, fg=secondary_color)
status_label.pack(pady=(10, 15))

# Button frame
button_frame = tk.Frame(main_frame, bg=bg_color)
button_frame.pack(pady=10)

# Modern-style buttons
load_button = tk.Button(
    button_frame, 
    text="โหลดรูปภาพ", 
    command=load_img, 
    font=button_font,
    bg=primary_color, 
    fg='white', 
    padx=15, 
    pady=8, 
    relief=tk.FLAT,
    activebackground='#2980b9',
    activeforeground='white',
    cursor="hand2"
)
load_button.pack(side=tk.LEFT, padx=10)

detect_button = tk.Button(
    button_frame, 
    text="ตรวจจับวัตถุ", 
    command=detect_obj, 
    font=button_font,
    bg=accent_color, 
    fg='white', 
    padx=15, 
    pady=8, 
    relief=tk.FLAT,
    activebackground='#16a085',
    activeforeground='white',
    cursor="hand2"
)
detect_button.pack(side=tk.LEFT, padx=10)

root.mainloop()