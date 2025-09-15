import os
import argparse
import pickle
import csv
import multiprocessing
from tqdm import tqdm

import lz4.frame
import snappy
import zlib
import zstandard as zstd

import torch
import numpy as np


def compress_data(data_bytes, algorithm, n_threads=0):
    if algorithm == 'lz4':
        return lz4.frame.compress(data_bytes, compression_level=1)
    elif algorithm == 'snappy':
        return snappy.compress(data_bytes)
    elif algorithm == 'zlib':
        return zlib.compress(data_bytes, level=1)
    elif algorithm == 'zstd':
        if n_threads > 0:
            cctx = zstd.ZstdCompressor(level=1, threads=n_threads)
        else:
            cctx = zstd.ZstdCompressor(level=1)
        return cctx.compress(data_bytes)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def get_compression_ratio(uncompressed_size, compressed_size):
    return (compressed_size / uncompressed_size) if uncompressed_size > 0 else 0


def collect_bfloat16_tensors(obj, out_list=None):
    if out_list is None:
        out_list = []

    if isinstance(obj, dict):
        for v in obj.values():
            collect_bfloat16_tensors(v, out_list)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            collect_bfloat16_tensors(x, out_list)
    elif isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            out_list.append(obj.view(torch.int16).flatten())

    return out_list


def bit_group_bfloat16(int16_tensor):
    numel = int16_tensor.numel()
    if numel == 0:
        return b''

    arr = int16_tensor.cpu().numpy()
    bits = (arr[:, None] >> np.arange(16)) & 1
    bits = bits.transpose()
    packed_planes = np.packbits(bits, axis=1, bitorder='little')
    return packed_planes.reshape(-1).tobytes()


def byte_group_bfloat16(int16_tensor):
    numel = int16_tensor.numel()
    if numel == 0:
        return b''

    arr = int16_tensor.cpu().numpy()
    arr_bytes = arr.view(np.uint8).reshape(-1, 2)
    low_bytes = arr_bytes[:, 0]
    high_bytes = arr_bytes[:, 1]
    grouped = np.concatenate([low_bytes, high_bytes], axis=0)
    return grouped.tobytes()


def difference_encode_bfloat16(int16_tensor, shift_val):
    if int16_tensor.numel() == 0:
        return torch.empty(0, dtype=torch.int16)

    float_data = int16_tensor.view(torch.bfloat16).float()
    diff_data = float_data - shift_val
    diff_bf16 = diff_data.to(torch.bfloat16)
    diff_int16 = diff_bf16.view(torch.int16)
    return diff_int16


def load_and_process_step0(file_path):
    with open(file_path, 'rb') as f:
        obj_step0 = pickle.load(f)
    
    bf16_tensors_0 = collect_bfloat16_tensors(obj_step0)
    
    if len(bf16_tensors_0) > 0:
        all_bf16_step0 = torch.cat(bf16_tensors_0, dim=0)
        float_data_step0 = all_bf16_step0.view(torch.bfloat16).float()
        float_data_np = float_data_step0.cpu().numpy()
        min_val = np.percentile(float_data_np, 1)
    else:
        all_bf16_step0 = torch.empty(0, dtype=torch.int16)
        min_val = 0.0
    
    return all_bf16_step0, min_val


def load_and_process_step1(file_path):
    with open(file_path, 'rb') as f:
        raw_data_step1 = f.read()
    original_size = len(raw_data_step1)

    with open(file_path, 'rb') as f:
        obj_step1 = pickle.load(f)
    
    bf16_tensors_1 = collect_bfloat16_tensors(obj_step1)
    
    if len(bf16_tensors_1) > 0:
        all_bf16_step1 = torch.cat(bf16_tensors_1, dim=0)
    else:
        all_bf16_step1 = torch.empty(0, dtype=torch.int16)
    
    return raw_data_step1, original_size, all_bf16_step1


def compute_compression_situations(all_bf16_step0, all_bf16_step1, min_val, raw_data_step1, original_size):
    bit_grouped_data = bit_group_bfloat16(all_bf16_step1)
    byte_grouped_data = byte_group_bfloat16(all_bf16_step1)
    bit_grouped_size = len(bit_grouped_data)
    byte_grouped_size = len(byte_grouped_data)

    if all_bf16_step1.numel() > 0:
        diff_min_int16 = difference_encode_bfloat16(all_bf16_step1, min_val)
        diff_min_data_bytes = diff_min_int16.numpy().tobytes()
        diff_min_bit_grouped_data = bit_group_bfloat16(diff_min_int16)
        diff_min_byte_grouped_data = byte_group_bfloat16(diff_min_int16)
        diff_min_data_size = len(diff_min_data_bytes)
        diff_min_bit_grouped_size = len(diff_min_bit_grouped_data)
        diff_min_byte_grouped_size = len(diff_min_byte_grouped_data)
    else:
        diff_min_data_bytes = b''
        diff_min_bit_grouped_data = b''
        diff_min_byte_grouped_data = b''
        diff_min_data_size = 0
        diff_min_bit_grouped_size = 0
        diff_min_byte_grouped_size = 0

    diff_int_data_bytes = b''
    diff_bf16_data_bytes = b''
    diff_int_size = 0
    diff_bf16_size = 0
    diff_int_bit_grouped_data = b''
    diff_int_byte_grouped_data = b''
    diff_bf16_bit_grouped_data = b''
    diff_bf16_byte_grouped_data = b''
    diff_int_bit_grouped_size = 0
    diff_int_byte_grouped_size = 0
    diff_bf16_bit_grouped_size = 0
    diff_bf16_byte_grouped_size = 0

    if all_bf16_step0.numel() == all_bf16_step1.numel() and all_bf16_step1.numel() > 0:
        difference_int16 = all_bf16_step1 - all_bf16_step0
        diff_int_data_bytes = difference_int16.numpy().tobytes()
        diff_int_size = len(diff_int_data_bytes)

        float_step0 = all_bf16_step0.view(torch.bfloat16).float()
        float_step1 = all_bf16_step1.view(torch.bfloat16).float()
        float_diff = float_step1 - float_step0
        bf16_diff = float_diff.to(torch.bfloat16)
        difference_bf16_int16 = bf16_diff.view(torch.int16)
        diff_bf16_data_bytes = difference_bf16_int16.numpy().tobytes()
        diff_bf16_size = len(diff_bf16_data_bytes)

        diff_int_bit_grouped_data = bit_group_bfloat16(difference_int16)
        diff_int_bit_grouped_size = len(diff_int_bit_grouped_data)
        diff_int_byte_grouped_data = byte_group_bfloat16(difference_int16)
        diff_int_byte_grouped_size = len(diff_int_byte_grouped_data)
        diff_bf16_bit_grouped_data = bit_group_bfloat16(difference_bf16_int16)
        diff_bf16_bit_grouped_size = len(diff_bf16_bit_grouped_data)
        diff_bf16_byte_grouped_data = byte_group_bfloat16(difference_bf16_int16)
        diff_bf16_byte_grouped_size = len(diff_bf16_byte_grouped_data)

    situations = [
        ('original',      raw_data_step1, original_size),
        ('bit_grouped',   bit_grouped_data, bit_grouped_size),
        ('byte_grouped',  byte_grouped_data, byte_grouped_size),
        ('diff_min_encoded',        diff_min_data_bytes, diff_min_data_size),
        ('diff_min_bit_grouped',    diff_min_bit_grouped_data, diff_min_bit_grouped_size),
        ('diff_min_byte_grouped',   diff_min_byte_grouped_data, diff_min_byte_grouped_size),
        ('diff_int',                diff_int_data_bytes, diff_int_size),
        ('diff_int_bit_grouped',    diff_int_bit_grouped_data, diff_int_bit_grouped_size),
        ('diff_int_byte_grouped',   diff_int_byte_grouped_data, diff_int_byte_grouped_size),
        ('diff_bfloat16',           diff_bf16_data_bytes, diff_bf16_size),
        ('diff_bfloat16_bit_grouped',  diff_bf16_bit_grouped_data, diff_bf16_bit_grouped_size),
        ('diff_bfloat16_byte_grouped', diff_bf16_byte_grouped_data, diff_bf16_byte_grouped_size),
    ]
    
    return situations


def compute_compression_results(situations, algorithms, n_threads_zstd):
    rows = []
    for (situation_name, uncompressed_bytes, uncompressed_sz) in situations:
        for algorithm in algorithms:
            row = {
                'algorithm': algorithm,
                'situation': situation_name,
                'uncompressed_size': uncompressed_sz,
                'compressed_size': None,
                'compression_ratio': None
            }
            if uncompressed_sz > 0:
                try:
                    comp_bytes = compress_data(
                        uncompressed_bytes, algorithm,
                        n_threads=n_threads_zstd
                    )
                    csize = len(comp_bytes)
                    row['compressed_size'] = csize
                    row['compression_ratio'] = get_compression_ratio(uncompressed_sz, csize)
                except Exception as e:
                    print(f"Compression failed: {algorithm} - {situation_name}: {e}")
            rows.append(row)
    
    return rows


def save_results_to_csv(rows, output_csv):
    csv_headers = ["algorithm", "situation", "uncompressed_size", "compressed_size", "compression_ratio"]
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_pair(file_step0, file_step1, output_csv, n_threads_zstd=0):
    algorithms = ['lz4', 'snappy', 'zlib', 'zstd']

    all_bf16_step0, min_val = load_and_process_step0(file_step0)
    raw_data_step1, original_size, all_bf16_step1 = load_and_process_step1(file_step1)
    
    situations = compute_compression_situations(
        all_bf16_step0, all_bf16_step1, min_val, raw_data_step1, original_size
    )
    
    rows = compute_compression_results(situations, algorithms, n_threads_zstd)
    save_results_to_csv(rows, output_csv)


def get_processing_tasks(dir_step0, dir_step1, output_dir, zstd_threads):
    step0_files = sorted([
        fn for fn in os.listdir(dir_step0)
        if os.path.isfile(os.path.join(dir_step0, fn))
    ])

    tasks = []
    for fn in step0_files:
        file_step0 = os.path.join(dir_step0, fn)
        if os.path.getsize(file_step0) < 2*1024*1024:
            continue
        file_step1 = os.path.join(dir_step1, fn)
        out_csv_path = os.path.join(output_dir, fn + ".csv")
        tasks.append((file_step0, file_step1, out_csv_path, zstd_threads))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_step0",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dir_step1",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--zstd_threads",
        type=int,
        default=8
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = get_processing_tasks(args.dir_step0, args.dir_step1, args.output_dir, args.zstd_threads)
    
    if len(tasks) == 0:
        print("No matching files found. Exiting.")
        return

    if args.num_workers > 1:
        with multiprocessing.Pool(args.num_workers) as pool:
            for _ in tqdm(pool.starmap(process_pair, tasks), total=len(tasks)):
                pass
    else:
        for task in tasks:
            process_pair(*task)


if __name__ == "__main__":
    main()