import threading
import tkinter as tk
import VGA_wrapped

def run_agent():
    VGA_wrapped.run()
    root.destroy()

root = tk.Tk()
root.title("VGA Agent Booting...")
root.geometry("400x150")
tk.Label(root, text="Launching the Visual Game Agent...", font=("Arial", 14)).pack(pady=30)
tk.Label(root, text="Please wait...", font=("Arial", 10)).pack()

threading.Thread(target=run_agent).start()
root.mainloop()
