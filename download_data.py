"""Download the fixed benchmark splits used in the paper."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = '925c80e387b814a7bd7e29406de1ca03cc54cc82'
BASE = 'https://raw.githubusercontent.com/sisinflab/Mind_the_Metric_SIGIR-26/' + COMMIT + '/'


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']


def fetch(entry, directory, verify_only):
    path = directory / entry['local_path']
    if not valid(path, entry):
        if verify_only:
            raise ValueError('Missing or modified dataset file: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.download')
        url = BASE + urllib.parse.quote(entry['path'], safe='/')
        try:
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
                for chunk in iter(lambda: response.read(2**20), b''):
                    stream.write(chunk)
            if not valid(temporary, entry):
                raise ValueError('Downloaded data checksum mismatch: ' + entry['path'])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return 'OK ' + entry['local_path']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('amazon-cds', 'douban', 'gowalla', 'yelp2018'), nargs='+')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'data/manifest.json').read_text())
    if args.dataset:
        entries = [entry for entry in entries if entry['dataset'] in args.dataset]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for message in pool.map(lambda entry: fetch(entry, args.data_dir, args.verify_only), entries):
            print(message, flush=True)


if __name__ == '__main__':
    main()
