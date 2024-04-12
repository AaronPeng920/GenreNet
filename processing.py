"""
Zhengwei Peng, 2024/2/15

This code is used to split all audio files in a folder or folder into several segments of the same length, 
discarding the last one that is less than the specified length.

USAGE: 
    python3 processing.py -i `file or dir path need to be processed` -o `save dir path` -c `config path`
"""

import os
import torchaudio
import argparse
import yaml

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help='A dir or file need to be processed.'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help='A dir to save processed files.'
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=30,
        help='segment duration'
    )
    parser.add_argument(
        "-s",
        "--resample_rate",
        type=int,
        default=16000,
        help='resample rate'
    )
    
    opt = parser.parse_args()
    return opt
    
def split_audio(input, output, segment_duration=30, resample_rate=16000):
    input_files = []
    file_count = 0
    segment_count = 0
    if os.path.isdir(input):
        for file in os.listdir(input):
            if file[0] == '.':
                continue
            input_files.append(os.path.join(input, file))
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise ValueError('input should be a file or dir.')
    
    if not os.path.isdir(output):
        raise ValueError('output should be a dir.')
        
    for input_file in input_files:
        waveform, ori_sample_rate = torchaudio.load(input_file)
        waveform = torchaudio.functional.resample(waveform, orig_freq=ori_sample_rate, new_freq=resample_rate)
        segment_samples = int(resample_rate * segment_duration)
        total_segments = waveform.size(1) // segment_samples


        for i in range(total_segments):
            segment = waveform[:, i * segment_samples : (i + 1) * segment_samples]
            output_file = os.path.join(output, f"{os.path.splitext(os.path.basename(input_file))[0]}_{i}.wav")
            torchaudio.save(output_file, segment, resample_rate)
            segment_count += 1
            
        file_count += 1
    
    print("A total of {} audio files were processed, generate {} segments of {} seconds.".format(file_count, segment_count, segment_duration))
    

if __name__ == "__main__":
    opt = get_opt()
    split_audio(opt.input, opt.output, segment_duration=opt.duration, resample_rate=opt.resample_rate)
