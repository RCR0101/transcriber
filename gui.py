import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import os
import pathlib
import sys
import threading
import importlib.util
from queue import Queue, Empty
import multiprocessing
import torch
import logging
import tempfile
from datetime import datetime

# Set up logging to file
log_file = os.path.join(tempfile.gettempdir(), 'transcriber_gui.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Starting application. Log file: {log_file}")
logger.info(f"Python version: {sys.version}")
logger.info(f"System platform: {sys.platform}")

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
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(os.path.expanduser(path))
        # Use pathlib for robust path handling
        norm_path = str(pathlib.Path(abs_path))
        logger.debug(f"Normalizing path: {path} -> {norm_path}")
        return norm_path
    except Exception as e:
        logger.error(f"Error normalizing path {path}: {e}")
        return path

def get_bundle_dir():
    """Get the directory where the application is running"""
    try:
        if getattr(sys, 'frozen', False):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            bundle_dir = sys._MEIPASS
            logger.info(f"Running in PyInstaller bundle: {bundle_dir}")
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
            logger.info(f"Running in development mode: {bundle_dir}")
        
        # Log the contents of the bundle directory
        logger.debug("Bundle directory contents:")
        for root, dirs, files in os.walk(bundle_dir):
            for name in files:
                logger.debug(f"  {os.path.join(root, name)}")
            for name in dirs:
                logger.debug(f"  {os.path.join(root, name)}/")
        
        return bundle_dir
    except Exception as e:
        logger.error(f"Error getting bundle directory: {e}")
        return os.path.dirname(os.path.abspath(__file__))

class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcriber GUI")
        
        # Store bundle directory
        self.bundle_dir = get_bundle_dir()
        
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
            norm_path = normalize_path(filename)
            logger.debug(f"Selected input file: {norm_path}")
            self.input_path.set(norm_path)
            # Set default output path
            input_path = pathlib.Path(norm_path)
            default_output = input_path.parent / f"{input_path.stem}.txt"
            self.output_path.set(normalize_path(str(default_output)))

    def select_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if filename:
            norm_path = normalize_path(filename)
            logger.debug(f"Selected output file: {norm_path}")
            self.output_path.set(norm_path)

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
                logger.error(f"Transcription error: {msg['text']}")
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
                self.transcribe_btn.config(state='normal')
                messagebox.showerror("Error", msg['text'])
        except Empty:
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.check_message_queue)

    def start_transcription(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_file:
            self.progress_var.set("Please select an input file")
            return
            
        if not output_file:
            # Use default output path if none specified
            input_path = pathlib.Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}.txt")
            self.output_path.set(normalize_path(output_file))
        
        # Log file paths
        logger.debug(f"Processing input file: {input_file}")
        logger.debug(f"Output file: {output_file}")
        
        # Verify files exist and are accessible
        if not os.path.exists(input_file):
            error_msg = f"Input file not found: {input_file}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.progress_var.set(f"Error: {error_msg}")
            return
        
        try:
            # Test if we can open the output file
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, 'a') as f:
                pass
        except Exception as e:
            error_msg = f"Cannot access output file: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.progress_var.set(f"Error: {error_msg}")
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
            # Add bundle directory to Python path
            if self.bundle_dir not in sys.path:
                sys.path.insert(0, self.bundle_dir)
                logger.debug(f"Added to sys.path: {self.bundle_dir}")
                logger.debug(f"Current sys.path: {sys.path}")

            # Import the transcriber module
            from transcriber.engine import WhisperEngine
            from transcriber.audio import load_audio

            # Initialize the engine and transcribe
            engine = WhisperEngine()
            logger.debug(f"Loading audio file: {input_file}")
            audio = load_audio(input_file)
            logger.debug("Audio loaded successfully")
            
            result = engine.transcribe(audio)
            logger.debug("Transcription complete")

            # Save the result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            logger.debug(f"Results saved to: {output_file}")

            # Signal completion
            self.message_queue.put({
                'type': 'complete',
                'text': f"Transcription complete! Saved to: {output_file}"
            })
        except Exception as e:
            logger.exception("Error during transcription")
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
    except Exception as e:
        logger.exception("Application error")
        messagebox.showerror("Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 