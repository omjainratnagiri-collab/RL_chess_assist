import argparse
import random
import time
from pathlib import Path
import sys
                

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from stockfish_stream_collate import stockfish_stream_collate
from stockfish_zst_dataset import StockfishZstDataset
from nnue import NNUE
from losses import value_loss_fn


class StreamingShuffleBuffer(IterableDataset):
    def __init__(self, dataset, buffer_size, seed=1):
        self.dataset = dataset
        self.buffer_size = int(buffer_size)
        self.seed = seed

    def __iter__(self):
        if self.buffer_size <= 1:
            yield from self.dataset
            return

        rng = random.Random(self.seed)
        buffer = []
        iterator = iter(self.dataset)

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass

        for item in iterator:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item

        rng.shuffle(buffer)
        yield from buffer


def soft_policy_loss(policy_logits, target_indices, target_probs):
    log_probs = F.log_softmax(policy_logits, dim=1)
    losses = []

    for row, (indices, probs) in enumerate(zip(target_indices, target_probs)):
        losses.append(-(log_probs[row, indices] * probs).sum())

    return torch.stack(losses).mean()


def train_batch(model, optimizer, device, batch):
    us_indices, them_indices, policy_indices, policy_probs, value_target = batch
    policy_indices = [item.to(device) for item in policy_indices]
    policy_probs = [item.to(device) for item in policy_probs]
    value_target = value_target.to(device)

    optimizer.zero_grad(set_to_none=True)
    policy_logits, value_pred = model(us_indices, them_indices)

    p_loss = soft_policy_loss(policy_logits, policy_indices, policy_probs)
    v_loss = value_loss_fn(value_pred, value_target)
    loss = p_loss + v_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    predictions = policy_logits.argmax(dim=1)
    top1 = sum(
        int(pred.item() == indices_[0].item())
        for pred, indices_ in zip(predictions, policy_indices)
    )

    return loss.item(), p_loss.item(), v_loss.item(), top1


def train_buffered_chunk(model, optimizer, device, buffered_batches, chunk_epochs):
    total_batches = 0
    total_positions = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0

    for _epoch in range(chunk_epochs):
        for batch in buffered_batches:
            batch_size = len(batch[2])                              
            loss, p_loss, v_loss, top1 = train_batch(model, optimizer, device, batch)
            total_batches += 1
            total_positions += batch_size
            total_loss += loss
            total_policy_loss += p_loss
            total_value_loss += v_loss
            total_top1 += top1

    return (
        total_positions, total_batches, total_loss,
        total_policy_loss, total_value_loss, total_top1
    )


def train_stream(
    model, dataloader, optimizer, device, args, checkpoint_dir, weights_out,
    start_chunk=0, best_loss=float("inf"), start_positions_seen=0
):
    model.train()

    chunk = start_chunk
    positions_seen = start_positions_seen
    chunk_positions = 0
    chunk_batches = 0
    chunk_loss = 0.0
    chunk_policy_loss = 0.0
    chunk_value_loss = 0.0
    chunk_top1 = 0
    chunk_start = time.time()
    buffered_batches = []

    for batch in dataloader:
        _us, _them, policy_indices, _probs, _value_target = batch
        batch_size = len(policy_indices)
        chunk_positions += batch_size
        positions_seen += batch_size

        if args.chunk_epochs == 1:
            loss, p_loss, v_loss, top1 = train_batch(model, optimizer, device, batch)
            chunk_batches += 1
            chunk_loss += loss
            chunk_policy_loss += p_loss
            chunk_value_loss += v_loss
            chunk_top1 += top1
        else:
            buffered_batches.append(batch)

        if chunk_positions >= args.positions_per_save:
            if args.chunk_epochs > 1:
                (trained_positions, chunk_batches, chunk_loss, chunk_policy_loss,
                 chunk_value_loss, chunk_top1) = train_buffered_chunk(
                    model, optimizer, device, buffered_batches, args.chunk_epochs
                )
            else:
                trained_positions = chunk_positions

            chunk += 1
            best_loss = save_chunk(
                chunk, positions_seen, chunk_positions, trained_positions,
                chunk_batches, chunk_loss, chunk_policy_loss, chunk_value_loss,
                chunk_top1, time.time() - chunk_start, model, optimizer,
                checkpoint_dir, weights_out, best_loss, args
            )
            chunk_positions = 0
            chunk_batches = 0
            chunk_loss = 0.0
            chunk_policy_loss = 0.0
            chunk_value_loss = 0.0
            chunk_top1 = 0
            buffered_batches = []
            chunk_start = time.time()

    if chunk_positions > 0:
        trained_positions = chunk_positions
        if args.chunk_epochs > 1:
            (trained_positions, chunk_batches, chunk_loss, chunk_policy_loss,
             chunk_value_loss, chunk_top1) = train_buffered_chunk(
                model, optimizer, device, buffered_batches, args.chunk_epochs
            )

        chunk += 1
        best_loss = save_chunk(
            chunk, positions_seen, chunk_positions, trained_positions,
            chunk_batches, chunk_loss, chunk_policy_loss, chunk_value_loss,
            chunk_top1, time.time() - chunk_start, model, optimizer,
            checkpoint_dir, weights_out, best_loss, args
        )

    if positions_seen == start_positions_seen:
        raise ValueError("No samples were produced.")

    return best_loss, chunk, positions_seen


def save_chunk(
    chunk, positions_seen, chunk_positions, trained_positions, chunk_batches,
    chunk_loss, chunk_policy_loss, chunk_value_loss, chunk_top1, elapsed,
    model, optimizer, checkpoint_dir, weights_out, best_loss, args
):
    loss = chunk_loss / chunk_batches
    p_loss = chunk_policy_loss / chunk_batches
    v_loss = chunk_value_loss / chunk_batches
    accuracy = 100.0 * chunk_top1 / trained_positions

    print()
    print("=" * 60)
    print(f"Chunk {chunk}")
    print("-" * 60)
    print(f"Chunk Positions : {chunk_positions}")
    print(f"Training Passes : {args.chunk_epochs}")
    print(f"Trained Samples : {trained_positions}")
    print(f"Total Positions : {positions_seen}")
    print(f"Total Loss      : {loss:.4f}")
    print(f"Policy Loss     : {p_loss:.4f}")
    print(f"Value Loss      : {v_loss:.4f}")
    print(f"Top-1 Accuracy  : {accuracy:.2f}%")
    print(f"Time            : {elapsed:.1f}s")
    print("=" * 60)

    checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}_{chunk}.pt"
    save_checkpoint(checkpoint_path, chunk, model, optimizer, loss, args,
                     positions_seen=positions_seen)
    print(f"Checkpoint saved: {checkpoint_path}")

    if loss < best_loss:
        best_loss = loss
        save_checkpoint(weights_out, chunk, model, optimizer, loss, args,
                         positions_seen=positions_seen)
        print(f"New best weights saved: {weights_out}")

    return best_loss


def save_checkpoint(path, chunk, model, optimizer, loss, args, positions_seen=0):
    torch.save(
        {
            "chunk": chunk,
            "epoch": chunk,
            "positions_seen": positions_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "args": vars(args),
        },
        path
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return (
            checkpoint.get("chunk", checkpoint.get("epoch", -1)),
            checkpoint.get("loss", float("inf")),
            checkpoint.get("positions_seen", 0)
        )

    model.load_state_dict(checkpoint)
    return 0, float("inf"), 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst-file", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--total-positions", type=int, default=None)
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--positions", type=int, default=None)
    parser.add_argument("--positions-per-save", type=int, default=500000)
    parser.add_argument("--chunk-epochs", type=int, default=1)
    parser.add_argument("--skip-positions", type=int, default=0)
    parser.add_argument("--max-pvs", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scan-all-depths", action="store_true")
    parser.add_argument("--policy-temperature", type=float, default=100.0)
    parser.add_argument("--value-scale", type=float, default=400.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--checkpoint-prefix", default="halfkp_stockfish_zst_chunk")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights-out", default="data/checkpoints/best_model_halfkp_stockfish_zst.pt")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    if args.chunk_epochs < 1:
        raise ValueError("--chunk-epochs must be >= 1")
    if args.shuffle_buffer < 0:
        raise ValueError("--shuffle-buffer must be >= 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"Using: {device}")

    model = NNUE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_out = Path(args.weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    start_chunk = 0
    positions_seen = 0
    best_loss = float("inf")
    resume_path = Path(args.resume) if args.resume else None

    if resume_path is not None and resume_path.exists():
        start_chunk, best_loss, positions_seen = load_checkpoint(
            resume_path, model, optimizer, device
        )
        print(f"Resuming from {resume_path} after chunk {start_chunk}.")
    else:
        print("Training from scratch.")

    total_positions_target = args.positions or args.total_positions or args.max_positions

    if args.start_position is not None:
        args.skip_positions = args.start_position
    else:
        args.skip_positions = 0

    print(
        "Stream settings: "
        f"epochs={args.epochs}, "
        f"start_position={args.skip_positions}, "
        f"total_positions={total_positions_target}, "
        f"positions_per_save={args.positions_per_save}, "
        f"shuffle_buffer={args.shuffle_buffer}, "
        f"max_pvs={args.max_pvs}, "
        f"use_first_eval={not args.scan_all_depths}"
    )

    for epoch in range(args.epochs):
        epoch_skip = args.skip_positions if epoch == 0 else 0
        dataset = StockfishZstDataset(
            zst_file=args.zst_file,
            max_depth=args.max_depth,
            max_positions=total_positions_target,
            skip_positions=epoch_skip,
            max_pvs=args.max_pvs,
            policy_temperature=args.policy_temperature,
            value_scale=args.value_scale,
            use_first_eval=not args.scan_all_depths
        )

        if args.shuffle_buffer > 1:
            dataset = StreamingShuffleBuffer(
                dataset, buffer_size=args.shuffle_buffer, seed=args.seed + epoch
            )

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "collate_fn": stockfish_stream_collate,
        }
        if args.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

        dataloader = DataLoader(**dataloader_kwargs)

        best_loss, start_chunk, positions_seen = train_stream(
            model, dataloader, optimizer, device, args, checkpoint_dir, weights_out,
            start_chunk=start_chunk, best_loss=best_loss, start_positions_seen=positions_seen
        )

    print()
    print("=" * 60)
    print("Training Finished")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Weights : {weights_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
