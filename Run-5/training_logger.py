"""
Comprehensive Training Logger for Netra-Adapt
Logs all metrics, visualizations, hyperparameters, and training progress
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentLogger:
    """
    Centralized logger for the entire Netra-Adapt pipeline.
    Creates timestamped directories and logs all important information.
    """
    
    def __init__(self, base_dir="logs"):
        """
        Initialize logger with timestamped run directory.
        
        Args:
            base_dir: Base directory for all logs (default: "logs")
        """
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(base_dir) / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each phase
        self.phase_dirs = {
            "source": self.run_dir / "01_source_training",
            "oracle": self.run_dir / "02_oracle_training",
            "adapt": self.run_dir / "03_adaptation",
            "evaluation": self.run_dir / "04_evaluation",
            "analysis": self.run_dir / "05_advanced_analysis"
        }
        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize run metadata
        self.metadata = {
            "run_id": self.timestamp,
            "start_time": datetime.now().isoformat(),
            "phases": {},
            "final_metrics": {}
        }
        
        # Create main log file
        self.log_file = self.run_dir / "experiment_log.txt"
        self._write_log("=" * 80)
        self._write_log(f"Netra-Adapt Experiment Run: {self.timestamp}")
        self._write_log("=" * 80)
        self._write_log("")
        
        print(f"\n📊 Experiment Logger Initialized")
        print(f"   Log Directory: {self.run_dir}")
        print(f"   Run ID: {self.timestamp}\n")
        
    def _write_log(self, message):
        """Write message to main log file and print to console."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    def log_phase_start(self, phase_name, hyperparameters):
        """
        Log the start of a training phase.
        
        Args:
            phase_name: Name of phase ("source", "oracle", "adapt")
            hyperparameters: Dictionary of hyperparameters
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self._write_log(f"\n{'='*80}")
        self._write_log(f"Phase Started: {phase_name.upper()}")
        self._write_log(f"Timestamp: {timestamp}")
        self._write_log(f"{'='*80}")
        self._write_log("\nHyperparameters:")
        for key, value in hyperparameters.items():
            self._write_log(f"  {key}: {value}")
        self._write_log("")
        
        # Store in metadata
        self.metadata["phases"][phase_name] = {
            "start_time": timestamp,
            "hyperparameters": hyperparameters,
            "epochs_run": 0,
            "best_loss": None,
            "final_loss": None,
            "training_time_seconds": 0,
            "early_stopped": False
        }
        
        # Save hyperparameters to phase directory
        phase_dir = self.phase_dirs[phase_name]
        with open(phase_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f, indent=4)
    
    def log_epoch(self, phase_name, epoch, total_epochs, metrics):
        """
        Log metrics for a single epoch.
        
        Args:
            phase_name: Name of phase ("source", "oracle", "adapt")
            epoch: Current epoch number
            total_epochs: Total number of epochs
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        # Log to main file
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self._write_log(f"Epoch {epoch}/{total_epochs}: {metrics_str}")
        
        # Save epoch metrics to CSV
        phase_dir = self.phase_dirs[phase_name]
        csv_file = phase_dir / "epoch_metrics.csv"
        
        # Create or append to CSV
        df_row = {"epoch": epoch, **metrics}
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
        else:
            df = pd.DataFrame([df_row])
        df.to_csv(csv_file, index=False)
        
        # Update metadata
        self.metadata["phases"][phase_name]["epochs_run"] = epoch
        self.metadata["phases"][phase_name]["final_loss"] = metrics.get("loss", None)
        if self.metadata["phases"][phase_name]["best_loss"] is None or \
           metrics.get("loss", float('inf')) < self.metadata["phases"][phase_name]["best_loss"]:
            self.metadata["phases"][phase_name]["best_loss"] = metrics.get("loss", None)
    
    def log_early_stopping(self, phase_name, epoch, best_loss):
        """
        Log early stopping event.
        
        Args:
            phase_name: Name of phase
            epoch: Epoch where early stopping occurred
            best_loss: Best loss achieved
        """
        self._write_log(f"\n⏹ EARLY STOPPING at Epoch {epoch}")
        self._write_log(f"   Best Loss: {best_loss:.4f}\n")
        
        self.metadata["phases"][phase_name]["early_stopped"] = True
        self.metadata["phases"][phase_name]["stopped_at_epoch"] = epoch
        self.metadata["phases"][phase_name]["best_loss"] = best_loss
    
    def log_phase_end(self, phase_name, training_time_seconds):
        """
        Log the end of a training phase.
        
        Args:
            phase_name: Name of phase
            training_time_seconds: Total training time in seconds
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hours = training_time_seconds / 3600
        minutes = (training_time_seconds % 3600) / 60
        
        self._write_log(f"\n{'='*80}")
        self._write_log(f"Phase Completed: {phase_name.upper()}")
        self._write_log(f"Timestamp: {timestamp}")
        self._write_log(f"Training Time: {hours:.1f}h {minutes:.1f}m ({training_time_seconds:.0f}s)")
        self._write_log(f"{'='*80}\n")
        
        self.metadata["phases"][phase_name]["end_time"] = timestamp
        self.metadata["phases"][phase_name]["training_time_seconds"] = training_time_seconds
        
        # Plot training curves
        self.plot_training_curves(phase_name)
    
    def plot_training_curves(self, phase_name):
        """
        Plot and save training loss curves.
        
        Args:
            phase_name: Name of phase
        """
        phase_dir = self.phase_dirs[phase_name]
        csv_file = phase_dir / "epoch_metrics.csv"
        
        if not csv_file.exists():
            return
        
        df = pd.read_csv(csv_file)
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["loss"], marker='o', linewidth=2, markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"{phase_name.capitalize()} Training - Loss Curve", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(phase_dir / "loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot additional metrics if available
        metric_cols = [col for col in df.columns if col not in ["epoch", "loss"]]
        if metric_cols:
            fig, axes = plt.subplots(1, len(metric_cols), figsize=(6*len(metric_cols), 5))
            if len(metric_cols) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, metric_cols):
                ax.plot(df["epoch"], df[metric], marker='o', linewidth=2, markersize=4, color='orange')
                ax.set_xlabel("Epoch", fontsize=12)
                ax.set_ylabel(metric.capitalize(), fontsize=12)
                ax.set_title(f"{metric.capitalize()} vs Epoch", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(phase_dir / "additional_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self._write_log(f"   Saved training curves to {phase_dir}/")
    
    def log_evaluation_metrics(self, model_name, metrics_dict):
        """
        Log evaluation metrics for a model.
        
        Args:
            model_name: Name of the model being evaluated
            metrics_dict: Dictionary of evaluation metrics
        """
        eval_dir = self.phase_dirs["evaluation"]
        
        self._write_log(f"\n{'='*80}")
        self._write_log(f"Evaluation: {model_name}")
        self._write_log(f"{'='*80}")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                self._write_log(f"  {metric}: {value:.4f}")
            else:
                self._write_log(f"  {metric}: {value}")
        self._write_log("")
        
        # Save to JSON
        json_file = eval_dir / f"{model_name.replace(' ', '_').replace('→', 'to')}_metrics.json"
        with open(json_file, "w") as f:
            json.dump(metrics_dict, f, indent=4, cls=NumpyEncoder)
        
        # Add to final metrics
        self.metadata["final_metrics"][model_name] = metrics_dict
    
    def log_visualization(self, phase, filename, description):
        """
        Log that a visualization was saved.
        
        Args:
            phase: Phase name ("evaluation", "analysis", etc.)
            filename: Name of saved file
            description: Description of the visualization
        """
        self._write_log(f"   Saved {description}: {filename}")
    
    def save_metadata(self):
        """Save all metadata to JSON file."""
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Calculate total training time
        total_time = sum([
            phase_data.get("training_time_seconds", 0) 
            for phase_data in self.metadata["phases"].values()
        ])
        self.metadata["total_training_time_seconds"] = total_time
        
        # Save to JSON
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    def generate_summary_report(self):
        """
        Generate a comprehensive markdown summary report.
        """
        self.save_metadata()
        
        report_path = self.run_dir / "EXPERIMENT_SUMMARY.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Netra-Adapt Experiment Summary\n\n")
            f.write(f"**Run ID:** `{self.timestamp}`\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            
            # Training Summary
            f.write(f"## 📊 Training Summary\n\n")
            
            total_time = self.metadata.get("total_training_time_seconds", 0)
            hours = total_time / 3600
            minutes = (total_time % 3600) / 60
            f.write(f"**Total Training Time:** {hours:.1f}h {minutes:.1f}m ({total_time:.0f}s)\n\n")
            
            # Phase-by-phase breakdown
            for phase_name, phase_data in self.metadata["phases"].items():
                f.write(f"### Phase: {phase_name.upper()}\n\n")
                
                # Hyperparameters
                f.write(f"**Hyperparameters:**\n```json\n")
                f.write(json.dumps(phase_data.get("hyperparameters", {}), indent=2))
                f.write(f"\n```\n\n")
                
                # Training statistics
                epochs_run = phase_data.get("epochs_run", 0)
                best_loss = phase_data.get("best_loss", None)
                final_loss = phase_data.get("final_loss", None)
                early_stopped = phase_data.get("early_stopped", False)
                training_time = phase_data.get("training_time_seconds", 0)
                
                f.write(f"**Training Statistics:**\n")
                f.write(f"- Epochs Run: {epochs_run}\n")
                if best_loss:
                    f.write(f"- Best Loss: {best_loss:.4f}\n")
                if final_loss:
                    f.write(f"- Final Loss: {final_loss:.4f}\n")
                f.write(f"- Early Stopped: {'Yes ⏹' if early_stopped else 'No (completed all epochs)'}\n")
                f.write(f"- Training Time: {training_time/60:.1f} minutes\n")
                f.write(f"\n**Loss Curve:**\n")
                f.write(f"![Loss Curve]({self.phase_dirs[phase_name].relative_to(self.run_dir)}/loss_curve.png)\n\n")
                f.write(f"---\n\n")
            
            # Evaluation Results
            if self.metadata["final_metrics"]:
                f.write(f"## 🎯 Evaluation Results\n\n")
                
                # Create comparison table
                f.write(f"### Metrics Comparison\n\n")
                
                # Get all unique metrics
                all_metrics = set()
                for metrics in self.metadata["final_metrics"].values():
                    all_metrics.update(metrics.keys())
                
                # Build table
                f.write("| Model | " + " | ".join(sorted(all_metrics)) + " |\n")
                f.write("|-------|" + "|".join(["-------" for _ in all_metrics]) + "|\n")
                
                for model_name, metrics in self.metadata["final_metrics"].items():
                    row = f"| {model_name} |"
                    for metric in sorted(all_metrics):
                        value = metrics.get(metric, "N/A")
                        if isinstance(value, float):
                            row += f" {value:.4f} |"
                        else:
                            row += f" {value} |"
                    f.write(row + "\n")
                
                f.write(f"\n---\n\n")
            
            # Files Generated
            f.write(f"## 📁 Generated Files\n\n")
            f.write(f"All outputs are organized in: `{self.run_dir}/`\n\n")
            
            f.write(f"### Directory Structure\n")
            f.write(f"```\n")
            f.write(f"run_{self.timestamp}/\n")
            f.write(f"├── experiment_log.txt          # Main log file\n")
            f.write(f"├── metadata.json               # Machine-readable metadata\n")
            f.write(f"├── EXPERIMENT_SUMMARY.md       # This file\n")
            f.write(f"├── 01_source_training/\n")
            f.write(f"│   ├── hyperparameters.json\n")
            f.write(f"│   ├── epoch_metrics.csv\n")
            f.write(f"│   └── loss_curve.png\n")
            f.write(f"├── 02_oracle_training/\n")
            f.write(f"│   └── ...\n")
            f.write(f"├── 03_adaptation/\n")
            f.write(f"│   └── ...\n")
            f.write(f"├── 04_evaluation/\n")
            f.write(f"│   └── *_metrics.json\n")
            f.write(f"└── 05_advanced_analysis/\n")
            f.write(f"    └── (visualizations from advanced_analysis.py)\n")
            f.write(f"```\n\n")
            
            # Footer
            f.write(f"---\n\n")
            f.write(f"*Generated by Netra-Adapt Training Logger*\n")
            f.write(f"*Timestamp: {datetime.now().isoformat()}*\n")
        
        self._write_log(f"\n{'='*80}")
        self._write_log(f"Summary Report Generated: {report_path}")
        self._write_log(f"{'='*80}\n")
        
        print(f"\n✅ Experiment Summary Generated: {report_path}")
        
        return report_path
    
    def get_phase_dir(self, phase_name):
        """Get the directory for a specific phase."""
        return self.phase_dirs.get(phase_name, self.run_dir)


# Singleton instance for easy access
_global_logger = None

def get_logger(base_dir="logs"):
    """Get or create the global experiment logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(base_dir)
    return _global_logger

def reset_logger():
    """Reset the global logger (useful for starting a new experiment)."""
    global _global_logger
    _global_logger = None
