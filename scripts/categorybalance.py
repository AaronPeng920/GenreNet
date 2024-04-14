import os
import shutil
import random
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help='A dir or file need to be processed.'
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help='A dir to save processed files.'
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=100,
        help='if n>0, remain files count in input_dir, else, move files count in input_dir'
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default='random',
        choices=['random', 'first', 'last'],
        help='move mode'
    )
    
    opt = parser.parse_args()
    return opt

def maintain_file_count(source_dir, target_dir, N, move_mode='random'):
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    if move_mode == 'random':
        random.shuffle(files)
    elif move_mode == 'last':
        files = files[::-1]

    if N > 0:
        if len(files) > N:
            num_files_to_move = len(files) - N
            os.makedirs(target_dir, exist_ok=True)
            for file in files[:num_files_to_move]:
                source_path = os.path.join(source_dir, file)
                target_path = os.path.join(target_dir, file)
                shutil.move(source_path, target_path)
    else:
        num_files_to_move = -N
        os.makedirs(target_dir, exist_ok=True)
        for file in files[:num_files_to_move]:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.move(source_path, target_path)
        
    
    print(f"{num_files_to_move} files has moved from {source_dir} to {target_dir}")

if __name__ == '__main__':
    opt = get_opt()
    maintain_file_count(opt.input_dir, opt.output_dir, opt.number, opt.mode)
