

import argparse
import io
from pathlib import Path

import zstandard as zstd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positions-per-chunk", type=int, default=5_000_000)
    parser.add_argument("--compression-level", type=int, default=9,
                         help="zstd compression level for the output chunks. "
                              "9 is a reasonable speed/size tradeoff; go lower "
                              "(e.g. 3) if you want this to run faster and "
                              "don't mind larger chunk files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=args.compression_level)

    chunk_index = 0
    lines_in_chunk = 0
    total_lines = 0
    out_file = None
    out_writer = None

    def open_new_chunk():
        nonlocal out_file, out_writer, chunk_index
        chunk_path = output_dir / f"chunk_{chunk_index}.jsonl.zst"
        out_file = open(chunk_path, "wb")
        out_writer = cctx.stream_writer(out_file)
        print(f"Writing {chunk_path} ...")

    open_new_chunk()

    with open(input_path, "rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                if not line.strip():
                    continue

                out_writer.write(line.encode("utf-8"))
                lines_in_chunk += 1
                total_lines += 1

                if total_lines % 500_000 == 0:
                    print(f"  ... {total_lines:,} positions processed")

                if lines_in_chunk >= args.positions_per_chunk:
                    out_writer.flush(zstd.FLUSH_FRAME)
                    out_writer.close()
                    out_file.close()
                    chunk_index += 1
                    lines_in_chunk = 0
                    open_new_chunk()

    if lines_in_chunk > 0:
        out_writer.flush(zstd.FLUSH_FRAME)
        out_writer.close()
        out_file.close()
    else:
                                                                      
        out_writer.close()
        out_file.close()
        empty_path = output_dir / f"chunk_{chunk_index}.jsonl.zst"
        if empty_path.exists() and empty_path.stat().st_size < 100:
            empty_path.unlink()

    print()
    print(f"Done. {total_lines:,} total positions split into "
          f"{chunk_index + (1 if lines_in_chunk > 0 else 0)} chunk files "
          f"in {output_dir}")
    print("Point --zst-file at chunk_N.jsonl.zst directly for future runs -- "
          "no --start-position/--skip-positions needed anymore, each chunk "
          "starts reading real data immediately.")


if __name__ == "__main__":
    main()