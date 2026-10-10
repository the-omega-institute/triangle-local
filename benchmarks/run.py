"""Run the complete sequential round-2 search recipe in one optimizer process."""
import argparse

from round2 import run as sweep
from refine import run as refine
from finish_search import run as finish
from continue_renou import run as continue_scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',default='benchmarks')
    p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--polish',type=int,default=200)
    a=p.parse_args()
    sweep(a.out_dir,a.seed,a.starts,a.steps,a.polish)
    refine(a.out_dir);finish(a.out_dir);continue_scan(a.out_dir)


if __name__=='__main__':main()
