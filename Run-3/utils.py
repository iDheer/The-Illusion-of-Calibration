import os
import matplotlib.pyplot as plt
import pandas as pd

class Logger:
    """Logger with automatic loss curve plotting"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(save_dir, "log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("epoch,loss\n")
            
    def log(self, epoch, loss):
        """Log metrics and generate loss curve plot"""
        with open(self.log_file, "a") as f:
            f.write(f"{epoch},{loss:.4f}\n")
        self._plot_loss_curve()
    
    def _plot_loss_curve(self):
        """Generate loss curve visualization"""
        try:
            df = pd.read_csv(self.log_file)
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['loss'], marker='o', linewidth=2, markersize=6)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(self.save_dir, "loss_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate loss curve plot: {e}")
