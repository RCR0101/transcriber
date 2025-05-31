import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
import pathlib

class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcriber GUI")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input file selection
        ttk.Label(main_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input).grid(row=0, column=2)
        
        # Output file selection
        ttk.Label(main_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output).grid(row=1, column=2)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Transcribe button
        ttk.Button(main_frame, text="Transcribe", command=self.transcribe).grid(row=3, column=0, columnspan=3, pady=10)

    def select_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Audio/Video Files", "*.mp3 *.mp4 *.wav *.m4a *.mov")]
        )
        if filename:
            self.input_path.set(filename)
            # Set default output path
            input_path = pathlib.Path(filename)
            default_output = input_path.parent / f"{input_path.stem}.txt"
            self.output_path.set(str(default_output))

    def select_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if filename:
            self.output_path.set(filename)

    def transcribe(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_file:
            self.progress_var.set("Please select an input file")
            return
            
        if not output_file:
            # Use default output path if none specified
            input_path = pathlib.Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}.txt")
            self.output_path.set(output_file)
        
        self.progress_var.set("Transcribing... Please wait")
        self.root.update()
        
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            transcriber_path = os.path.join(script_dir, "dist", "transcriber")
            
            # Run the transcriber
            process = subprocess.run(
                [transcriber_path, input_file, "-o", output_file],
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                self.progress_var.set(f"Transcription complete! Saved to: {output_file}")
            else:
                self.progress_var.set(f"Error: {process.stderr}")
        except Exception as e:
            self.progress_var.set(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = TranscriberGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 