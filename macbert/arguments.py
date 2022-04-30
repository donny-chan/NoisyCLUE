import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', required=True)

    # Hyperparameters
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--grad_acc_steps', type=int, default=16)

    # Logging
    p.add_argument('--log_interval', type=int, default=5)
    p.add_argument('--tqdm', action='store_true')

    return p.parse_args()