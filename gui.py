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

# Set up logging
log_dir = tempfile.gettempdir()
log_file = os.path.join(log_dir, f'transcriber_gui_{datetime.now():%Y%m%d_%H%M%S}.log')
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

# Configure PyTorch for single-threaded operation
torch.set_num_threads(1)
os.environ.update({
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1"
})

def normalize_path(path: str) -> str:
    """Convert path to absolute and normalize for platform compatibility."""
    if not path:
        return path
    try:
        return str(pathlib.Path(os.path.expanduser(path)).resolve())
    except Exception as e:
        logger.error(f"Path normalization failed: {e}", exc_info=True)
        return path

def get_bundle_dir() -> str:
    """Get the application's bundle directory."""
    try:
        if getattr(sys, 'frozen', False):
            bundle_dir = sys._MEIPASS
            logger.info(f"Running from PyInstaller bundle: {bundle_dir}")
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
            logger.info(f"Running in development mode: {bundle_dir}")
        return bundle_dir
    except Exception as e:
        logger.error(f"Failed to get bundle directory: {e}", exc_info=True)
        return os.path.dirname(os.path.abspath(__file__))

class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcriber")
        self.bundle_dir = get_bundle_dir()
        
        # Initialize transcription process tracking
        self.transcription_thread = None
        self.is_transcribing = False
        
        self.setup_gui()
        self.message_queue = Queue()
        self.check_message_queue()
        logger.info("GUI initialized")

    def setup_gui(self):
        """Set up the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Input file selection
        ttk.Label(main_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
        self.input_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input).grid(row=0, column=2)

        # Output file selection
        ttk.Label(main_frame, text="Output File:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output).grid(row=1, column=2)

        # Progress display
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=2, column=0, columnspan=3, pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        self.progress_bar.grid_remove()

        # Transcribe button
        self.transcribe_btn = ttk.Button(main_frame, text="Transcribe", command=self.start_transcription)
        self.transcribe_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # Configure grid weights
        for i in range(3):
            main_frame.columnconfigure(i, weight=1 if i == 1 else 0)

    def select_input(self):
        """Handle input file selection."""
        filename = filedialog.askopenfilename(
            title="Select Audio/Video File",
            filetypes=[("Audio/Video Files", "*.mp3 *.mp4 *.wav *.m4a *.mov")]
        )
        if filename:
            norm_path = normalize_path(filename)
            logger.info(f"Selected input file: {norm_path}")
            self.input_path.set(norm_path)
            
            # Set default output path
            output = pathlib.Path(norm_path).with_suffix('.txt')
            self.output_path.set(str(output))

    def select_output(self):
        """Handle output file selection."""
        filename = filedialog.asksaveasfilename(
            title="Save Transcript As",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if filename:
            norm_path = normalize_path(filename)
            logger.info(f"Selected output file: {norm_path}")
            self.output_path.set(norm_path)

    def check_message_queue(self):
        """Process messages from the transcription thread."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                msg_type = msg.get('type', '')
                msg_text = msg.get('text', '')
                
                if msg_type == 'progress':
                    self.progress_var.set(msg_text)
                    logger.info(f"Progress: {msg_text}")
                elif msg_type == 'complete':
                    self.progress_var.set(msg_text)
                    self.progress_bar.stop()
                    self.progress_bar.grid_remove()
                    self.transcribe_btn.config(state='normal')
                    self.is_transcribing = False
                    logger.info("Transcription complete")
                elif msg_type == 'error':
                    self.progress_var.set(f"Error: {msg_text}")
                    self.progress_bar.stop()
                    self.progress_bar.grid_remove()
                    self.transcribe_btn.config(state='normal')
                    self.is_transcribing = False
                    logger.error(f"Transcription error: {msg_text}")
                    messagebox.showerror("Error", msg_text)
        except Empty:
            pass
        finally:
            self.root.after(100, self.check_message_queue)

    def validate_paths(self) -> bool:
        """Validate input and output paths."""
        input_file = self.input_path.get()
        output_file = self.output_path.get()

        if not input_file:
            messagebox.showerror("Error", "Please select an input file")
            return False

        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return False

        try:
            output_dir = os.path.dirname(output_file) if output_file else None
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Cannot access output location: {e}")
            return False

    def start_transcription(self):
        """Start the transcription process."""
        # Prevent multiple transcription processes
        if self.is_transcribing:
            logger.warning("Transcription already in progress")
            return

        if not self.validate_paths():
            return

        self.is_transcribing = True
        self.transcribe_btn.config(state='disabled')
        self.progress_bar.grid()
        self.progress_bar.start(10)
        self.progress_var.set("Transcribing... Please wait")

        # Start transcription in a new thread
        self.transcription_thread = threading.Thread(
            target=self.run_transcription,
            args=(self.input_path.get(), self.output_path.get()),
            daemon=True
        )
        self.transcription_thread.start()
        logger.info("Started transcription thread")

    def run_transcription(self, input_file: str, output_file: str):
        """Run the transcription process in a separate thread."""
        try:
            # Add bundle directory to Python path
            if self.bundle_dir not in sys.path:
                sys.path.insert(0, self.bundle_dir)

            from transcriber.engine import WhisperEngine
            
            # Initialize engine and transcribe
            engine = WhisperEngine()
            logger.info(f"Processing file: {input_file}")
            result = engine.transcribe(input_file)

            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            logger.info(f"Saved transcript to: {output_file}")

            self.message_queue.put({
                'type': 'complete',
                'text': f"Transcription complete! Saved to: {output_file}"
            })

        except Exception as e:
            logger.error("Transcription failed", exc_info=True)
            self.message_queue.put({
                'type': 'error',
                'text': str(e)
            })

def main():
    """Start the application."""
    # Enable multiprocessing support for PyInstaller
    multiprocessing.freeze_support()
    
    try:
        root = tk.Tk()
        root.title("Audio Transcriber")
        
        # Prevent multiple instances
        app = TranscriberGUI(root)
        
        # Configure window
        root.protocol("WM_DELETE_WINDOW", root.quit)  # Handle window close properly
        root.mainloop()
    except Exception as e:
        logger.critical("Application failed to start", exc_info=True)
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main() 