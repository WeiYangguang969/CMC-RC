"""Train CMR-RC and evaluate full-catalog recommendations."""
import argparse
import json
from pathlib import Path
import time

from model import base_key, fit_bank, key, load_model, operators, score, sha, write_json
from dataloader import Dataset
from metrics import metric_values
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATASETS = ('amazon-cds', 'douban', 'gowalla', 'yelp2018')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--config', type=Path, help='Default: configs/<dataset>.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reg', type=float, help='Override ridge strength (lambda)')
    parser.add_argument('--nu', type=float, help='Override conditional-moment strength')
    parser.add_argument('--gamma', type=float, help='Override user residual weight')
    parser.add_argument('--beta', type=float, help='Override item residual weight')
    parser.add_argument('--k', type=int, help='Override the number of local neighbors')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = json.loads((args.config or ROOT / 'configs' / (args.dataset + '.json')).read_text())
    for argument, name in [('reg', 'lambda'), ('nu', 'nu'), ('gamma', 'gamma'), ('beta', 'beta'), ('k', 'k')]:
        value = getattr(args, argument)
        if value is not None:
            config[name] = value
    if args.nu is not None:
        config['statistic'] = ('none' if args.nu == 0 else
                               'uncentered' if config['statistic'] == 'none' else config['statistic'])
    config['candidate_pool'] = max(config['candidate_pool'], config['k'])
    if config['k'] < 1 or config['lambda'] <= 0:
        parser.error('k and ridge strength must be positive')

    ds = Dataset(args.dataset, args.data_dir)
    # Keep different training inputs, configurations and numerical environments
    # in separate caches, including when running ablations.
    import numpy as np
    import scipy
    import sklearn
    environment = {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    run_id = key(dict(config=config, train=sha(ds.base / 'train.tsv'), environment=environment))
    cache = args.cache_dir / args.dataset / run_id
    output = args.output_dir / args.dataset / run_id
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        fit_bank(ds, [config], cache / 'models', args.workers)
        P, V, _, _ = operators(ds, cache / 'operators', args.workers,
                              user=bool(config['gamma']),
                              spectral=bool(config['beta'] or (config['gamma'] and config['delta'] != 1)),
                              rank=config['rank'])
        B, scale = load_model(cache / 'models' / (base_key(config) + '.npz'))
        top = score(ds, B, scale, P, V, [config], workers=args.workers)[0][0]
        values, _, valid = metric_values(top, ds.labels(args.split))

    result = dict(dataset=args.dataset, split=args.split, config=config, metrics=values,
                  evaluated_users=int(valid.sum()), seconds=time.perf_counter() - started)
    write_json(output / ('metrics_' + args.split + '.json'), result)
    with (output / 'top20.tsv').open('w', encoding='utf-8') as stream:
        stream.write('user_id\trank\titem_id\n')
        for user, items in zip(ds.users, ds.items[top]):
            for rank, item in enumerate(items, 1):
                stream.write(f'{user}\t{rank}\t{item}\n')
    print(json.dumps(result, indent=2))
    print('Results:', output.resolve())


if __name__ == '__main__':
    main()
