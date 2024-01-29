from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_mnist_data, RunningMeanStd


def main(args):
    if args.debug_yes:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=12346, stdoutToServer=True, stderrToServer=True, suspend=False)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_test, x_train, y_test, y_train = get_mnist_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    nail_size = x_train.shape[-1]

    scaler = RunningMeanStd(shape=(nail_size,))
    scaler.update(x_train)
    train_dataset = ThroughDataset(scaler.normalize(x_train), y_train[:, None])
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=0)

    valid_dataset = ThroughDataset(scaler.normalize(x_test), y_test[:, None])
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, num_workers=0)

    model = nn.Sequential(
        MLPSkipper(nail_size, 256, 256),
        nn.Dropout(),
        nn.ReLU(),
        MLPSkipper(256, 256, 256),
        nn.Dropout(),
        nn.ReLU(),
        MLPSkipper(256, 256, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    writer = SummaryWriter(log_dir=f"{proj_dir}/logdir/{args.eval_name}")

    global_step = 0
    model.to("cuda")
    for ep in tqdm(range(1000)):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to("cuda", dtype=torch.float)
            y = y.to("cuda", dtype=torch.long).reshape(-1)
            y_hat = model.forward(x)

            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print(f"Epoch {ep} loss: {loss.item()}")
        writer.add_scalar("train/loss", loss.item(), global_step)

        # Validation loss
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_dataloader):
                x = x.to("cuda", dtype=torch.float)
                y = y.to("cuda", dtype=torch.long).reshape(-1)
                y_hat = model.forward(x)

                loss = torch.nn.functional.cross_entropy(y_hat, y)
                print(f"Validation loss: {loss.item()}")
                break
        writer.add_scalar("eval/loss", loss.item(), global_step)

        with torch.no_grad():
            all_accs = []
            for i, (x, y) in enumerate(valid_dataloader):
                x = x.to("cuda", dtype=torch.float)
                y = y.to("cuda", dtype=torch.long).reshape(-1)
                y_hat = model.forward(x)
                y_hat = y_hat.argmax(dim=-1)

                acc = (y_hat == y).float().mean()
                print(f"Validation acc: {acc.item()}")
                all_accs.append(acc)
        writer.add_scalar("eval/token_0_acc", acc.mean().item(), global_step)
        model.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug_yes", "-d", action="store_true")  # if set, will pause the program
    parser.add_argument("--jab_yes", "-j", action="store_true")
    # parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--eval_name", type=str, required=True)
    parser.add_argument("--base_name", type=str)  # Optional, if set, will load the snapshot and continue training
    # parser.add_argument("--batch_size", type=int, required=True)
    # parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    main(args)
