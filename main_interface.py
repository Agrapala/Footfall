import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import subprocess
import sys
import os

# Helper to get python executable
PYTHON_EXEC = sys.executable

# Script names
SCRIPTS = {
    'Age & Gender Detection': 'age_gender.py',
    'Unique Human Counting': 'count.py',
    'Motion Tracking': 'motion.py',
    'Face Recognition Security': 'thieves.py',
}

def run_count():
    # Dialog for method selection
    method = simpledialog.askstring(
        "Counting Method",
        "Choose counting method:\n1. Face Recognition (more accurate)\n2. Body Detection (faster)\nEnter 1 or 2:",
        parent=root
    )
    if method not in ('1', '2'):
        messagebox.showerror("Error", "Invalid method selected.")
        return
    # Launch count.py with method as argument (simulate input)
    # We'll use subprocess and pass input via stdin
    script_path = os.path.join(os.path.dirname(__file__), SCRIPTS['Unique Human Counting'])
    try:
        # Pass the method as input to the script
        subprocess.Popen([PYTHON_EXEC, script_path], stdin=subprocess.PIPE).communicate(input=f"{method}\n".encode())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run count.py: {e}")

def run_thieves():
    # Dialog for mode selection
    mode = simpledialog.askstring(
        "Face Recognition Security",
        "Choose mode:\nadd - Add Known Person\ndetect - Detect Faces in Image\nmonitor - Start Video Monitoring\nlist - List Known Faces\nremove - Remove Known Face\nEnter mode:",
        parent=root
    )
    if mode not in ("add", "detect", "monitor", "list", "remove"):
        messagebox.showerror("Error", "Invalid mode selected.")
        return
    args = [PYTHON_EXEC, os.path.join(os.path.dirname(__file__), SCRIPTS['Face Recognition Security'])]
    args += ["--mode", mode]
    # Collect additional info as needed
    if mode == "add":
        name = simpledialog.askstring("Add Person", "Enter person name:", parent=root)
        if not name:
            messagebox.showerror("Error", "Name required.")
            return
        image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not image_path:
            messagebox.showerror("Error", "Image file required.")
            return
        args += ["--name", name, "--image", image_path]
    elif mode == "detect":
        image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not image_path:
            messagebox.showerror("Error", "Image file required.")
            return
        args += ["--image", image_path]
        save = messagebox.askyesno("Save Result", "Save detection result image?")
        if save:
            args.append("--save")
    elif mode == "monitor":
        save = messagebox.askyesno("Save Detections", "Save detection log?")
        if save:
            args.append("--save")
    elif mode == "remove":
        name = simpledialog.askstring("Remove Person", "Enter person name to remove:", parent=root)
        if not name:
            messagebox.showerror("Error", "Name required.")
            return
        args += ["--name", name]
    # For 'list', no extra args
    try:
        subprocess.Popen(args)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run thieves.py: {e}")

def run_script(script_name):
    if script_name == SCRIPTS['Unique Human Counting']:
        run_count()
    elif script_name == SCRIPTS['Face Recognition Security']:
        run_thieves()
    else:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found: {script_name}")
            return
        try:
            subprocess.Popen([PYTHON_EXEC, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {script_name}: {e}")

# Create main window
root = tk.Tk()
root.title("Footfall Unified Interface")
root.geometry("400x400")
root.resizable(False, False)

# Title label
title = tk.Label(root, text="Footfall Unified Interface", font=("Arial", 18, "bold"), pady=20)
title.pack()

# Add buttons for each script
for label, script in SCRIPTS.items():
    btn = tk.Button(root, text=label, font=("Arial", 14), width=30, height=2,
                    command=lambda s=script: run_script(s))
    btn.pack(pady=8)

# Quit button
quit_btn = tk.Button(root, text="Quit", font=("Arial", 12), width=15, command=root.quit, bg="#d9534f", fg="white")
quit_btn.pack(pady=20)

root.mainloop() 