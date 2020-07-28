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
        'module', choices=list(MODULES.keys()),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    MODULES[args.module].run_test()


if __name__ == '__main__':
    main()
