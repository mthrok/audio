import argparse

import save_audio_file


MODULES = {mod.__name__: mod for mod in [save_audio_file]}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'module', choices=list(MODULES.keys()),
    )
    return parser.parse_args()


def run_test(module_name):
    print(f'Testing {module_name}')
    module = MODULES[module_name]
    print('Testing Torchscript')
    elapsed = module.test_torchscript()
    print('Elapsed: {elapsed} seconds')
    print('Testing pybind11')
    elapsed = module.test_pybind11()
    print('Elapsed: {elapsed} seconds')


def main():
    args = _parse_args()
    run_test(args.module)


if __name__ == '__main__':
    main()
