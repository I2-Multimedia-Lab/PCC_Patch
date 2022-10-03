import os
import argparse
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud

import pc_io
parser = argparse.ArgumentParser(
    prog='eval.py',
    description='Eval decompressed point clouds PSNR and bitrate.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('pc_error_path', help='Path to pc_error executable.')
parser.add_argument('input_path', help='Point clouds glob pattern to be compressed.')
parser.add_argument('compressed_path', help='Compressed point cloud files.')
parser.add_argument('decompressed_path', help='Decompressed point clouds from comporessed files.')
parser.add_argument('output_eval_file', help='Excel .csv file to show the evaluation detail of each compressed point cloud.')

args = parser.parse_args()

# CALC PSNR BETWEEN FILE AND DECOMPRESSED FILE
def pc_error(f, df):
    command = f'{args.pc_error_path} -a {f} -b {df} --knn {32}'
    #print("Executing " + command)
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    decoded_output = output.decode('utf-8').split('\n')
    data_lines = [x for x in decoded_output if '   ### ' in x]
    parsed_data_lines = [x[len('   ### '):] for x in data_lines]
    # Before last value : information about the metric
    # Last value : metric value
    data = [(','.join(x[:-1]), x[-1]) for x in [x.split(',') for x in parsed_data_lines]]
    return data

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

# GET FILE NAME FROM DECOMPRESSED PATH
files = pc_io.get_files(args.input_path)
files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])

filenames = np.array([os.path.split(x)[1] for x in files])

# .csv COLUMNS: [filename, p2pointPSNR, p2planePSNR, n_points_input, n_points_output, bpp]
ipt_files, p2pointPSNRs, p2planePSNRs, n_points_inputs, n_points_outputs, bpps = [], [], [], [], [], []

print('Evaluating...')
for i in tqdm(range(len(filenames))):
    input_f = files[i]
    comp_s_f = os.path.join(args.compressed_path, filenames[i] + '.s.bin')
    comp_h_f = os.path.join(args.compressed_path, filenames[i] + '.h.bin')
    comp_p_f = os.path.join(args.compressed_path, filenames[i] + '.p.bin')
    decomp_f = os.path.join(args.decompressed_path, filenames[i] + '.dec.ply')

    if not os.path.exists(decomp_f):
        continue

    ipt_files.append(filenames[i])
    # GET PSNR
    data = pc_error(input_f, decomp_f)
    p2pointPSNRs.append(round(float(data[-3][1]), 3))
    p2planePSNRs.append(round(float(data[-1][1]), 3))
    # GET NUMBER OF POINTS
    n_points_input = get_n_points(input_f)
    n_points_output = get_n_points(decomp_f)
    n_points_inputs.append(n_points_input)
    n_points_outputs.append(n_points_output)
    # GET BPP
    bpp = (get_file_size_in_bits(comp_s_f) + get_file_size_in_bits(comp_h_f) + get_file_size_in_bits(comp_p_f)) / n_points_input
    bpps.append(bpp)

# SAVE AS AN EXCEL .csv
df = pd.DataFrame()
df['filename'] = ipt_files
df['p2pointPSNR'] = p2pointPSNRs
df['p2planePSNR'] = p2planePSNRs
df['n_points_input'] = n_points_inputs
df['n_points_output'] = n_points_outputs
df['bpp'] = bpps
df.to_csv(args.output_eval_file)

print(f'Done! The average p2pointPSNR: {round(np.array(p2pointPSNRs).mean(), 3)} | p2plane PSNR: {round(np.array(p2planePSNRs).mean(), 3)} | bpp: {round(np.array(bpps).mean(), 3)}')
