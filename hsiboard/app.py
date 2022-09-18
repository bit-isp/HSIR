import argparse
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSIR Board')
    parser.add_argument('--logdir', default='results')
    args = parser.parse_args()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
    subprocess.call(['streamlit', 'run', path, '--', '--logdir', args.logdir])
