import argparse

import save_audio_file


MODULES = {mod.__name__: mod for mod in [save_audio_file]}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'module', choices=list(MODULES.keys()),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    MODULES[args.module].run_test()


if __name__ == '__main__':
    main()
