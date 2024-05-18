import os
from huggingface_hub import snapshot_download
from huggingface_hub import login
import argparse
login()

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--cache_path', type=str, default='/tmp')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
    snapshot_download(
        repo_id=args.hf_path, 
        local_dir=args.save_path, 
        local_dir_use_symlinks=False,
        force_download=True,
        cache_dir=args.cache_path)
    print(f'Download {args.hf_path} to {args.save_path} is done.')
else:
    print(f'{args.save_path} already exists.')
