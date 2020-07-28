import argparse

import save_audio_file
import load_audio_file
import apply_effects_tensor


MODULES = {
    mod.__name__: mod for mod in
    [save_audio_file, load_audio_file, apply_effects_tensor]
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--duration', type=float, default=1.0,
        help='Duration of audio data/file.',
    )
    parser.add_argument(
        '--num-trials', type=int, default=1000,
        help='The number of times to run the target function.',
    )
    parser.add_argument(
        '--module', choices=list(MODULES.keys()),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    print(args)
    modules = MODULES.values() if args.module is None else [MODULES[args.module]]
    for module in modules:
        module.run_test(num_trials=args.num_trials, duration=args.duration)


if __name__ == '__main__':
    main()
