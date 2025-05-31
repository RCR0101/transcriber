import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
import pathlib
import sys
import threading
import importlib.util
from queue import Queue, Empty
import multiprocessing
import torch

# Force PyTorch to use only one thread to prevent multiple instances
torch.set_num_threads(1)

# Ensure numpy only uses one thread
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def normalize_path(path):
    """Normalize path for Windows compatibility"""
    if not path:
        return path
    norm_path = os.path.abspath(os.path.expanduser(path))
    return norm_path.replace('/', '\\') if sys.platform == 'win32' else norm_path

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
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.progress_bar.grid_remove()  # Hide initially
        
        # Transcribe button
        self.transcribe_btn = ttk.Button(main_frame, text="Transcribe", command=self.start_transcription)
        self.transcribe_btn.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Message queue for thread communication
        self.message_queue = Queue()
        self.check_message_queue()

    def select_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Audio/Video Files", "*.mp3 *.mp4 *.wav *.m4a *.mov")]
        )
        if filename:
            # Normalize the path for Windows compatibility
            norm_path = normalize_path(filename)
            self.input_path.set(norm_path)
            # Set default output path
            input_path = pathlib.Path(filename)  # Use original path for pathlib
            default_output = input_path.parent / f"{input_path.stem}.txt"
            self.output_path.set(normalize_path(str(default_output)))

    def select_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if filename:
            self.output_path.set(normalize_path(filename))

    def check_message_queue(self):
        """Check for messages from the transcription thread"""
        try:
            msg = self.message_queue.get_nowait()
            if msg.get('type') == 'progress':
                self.progress_var.set(msg['text'])
            elif msg.get('type') == 'complete':
                self.progress_var.set(msg['text'])
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
                self.transcribe_btn.config(state='normal')
            elif msg.get('type') == 'error':
                self.progress_var.set(f"Error: {msg['text']}")
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
                self.transcribe_btn.config(state='normal')
        except Empty:
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.check_message_queue)

    def start_transcription(self):
        input_file = self.input_path.get().strip('"')  # Remove quotes for processing
        output_file = self.output_path.get().strip('"')  # Remove quotes for processing
        
        if not input_file:
            self.progress_var.set("Please select an input file")
            return
            
        if not output_file:
            # Use default output path if none specified
            input_path = pathlib.Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}.txt")
            self.output_path.set(normalize_path(output_file))
        
        # Verify files exist and are accessible
        if not os.path.exists(input_file):
            self.progress_var.set(f"Error: Input file not found: {input_file}")
            return
        
        try:
            # Test if we can open the output file
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, 'a') as f:
                pass
        except Exception as e:
            self.progress_var.set(f"Error: Cannot access output file: {str(e)}")
            return
        
        # Disable the transcribe button and show progress
        self.transcribe_btn.config(state='disabled')
        self.progress_bar.grid()
        self.progress_bar.start(10)
        self.progress_var.set("Transcribing... Please wait")
        
        # Start transcription in a separate thread
        thread = threading.Thread(
            target=self.run_transcription,
            args=(input_file, output_file),
            daemon=True
        )
        thread.start()

    def run_transcription(self, input_file, output_file):
        try:
            # Get the directory where this script is located
            if getattr(sys, 'frozen', False):
                # If we're running in a bundle
                script_dir = sys._MEIPASS
            else:
                # If we're running in development
                script_dir = os.path.dirname(os.path.abspath(__file__))

            # Add the script directory to Python path so we can import transcriber
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            # Import the transcriber module
            from transcriber.engine import WhisperEngine
            from transcriber.audio import load_audio

            # Initialize the engine and transcribe
            engine = WhisperEngine()
            audio = load_audio(input_file)
            result = engine.transcribe(audio)

            # Save the result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])

            # Signal completion
            self.message_queue.put({
                'type': 'complete',
                'text': f"Transcription complete! Saved to: {output_file}"
            })
        except Exception as e:
            # Signal error
            self.message_queue.put({
                'type': 'error',
                'text': str(e)
            })

def main():
    try:
        # Ensure only one instance runs
        multiprocessing.freeze_support()
        
        # Create and run the GUI
        root = tk.Tk()
        # Set window title
        root.title("Transcriber")
        # Try to bring window to front
        root.lift()
        root.attributes('-topmost', True)
        root.attributes('-topmost', False)
        
        app = TranscriberGUI(root)
        root.mainloop()
    except PermissionError as e:
        # Show error dialog if we can't access files
        import tkinter.messagebox as messagebox
        messagebox.showerror("Permission Error", 
            "Cannot access required files. Try running as administrator or moving to a different folder.")
        sys.exit(1)
    except Exception as e:
        import tkinter.messagebox as messagebox
        messagebox.showerror("Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 