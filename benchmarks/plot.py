"""Standalone scientific figures from the saved, certified CSV measurements."""
import csv
from triangle_local.io import load_csv, dump_csv
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out=Path(__file__).resolve().parent
    def read(path):
        return load_csv(path)
    rows=read(out/'best-fits.csv');old=read(out/'round1'/'best-fits.csv')
    def curve(data,name):
        selected=sorted([r for r in data if r['family']==name],key=lambda r:float(Fraction(r['parameter'])))
        return np.array([float(Fraction(r['parameter'])) for r in selected]),np.array([float(r.get('L2_rational',r['L2_float'])) for r in selected]),selected
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    settings=[('elegant-source','EJM: independent source noise',.8),
              ('elegant-detector','EJM: independent detector noise',.86),
              ('fritz-visibility','Fritz: source visibility',1/np.sqrt(2)),
              ('renou-u2',r'Renou: noiseless $u^2$ scan',None),
              ('renou-source',r'Renou, $u^2=0.85$: source noise',.89),
              ('renou-detector',r'Renou, $u^2=0.85$: detector noise',.91)]
    for ax,(name,title,threshold) in zip(axs.ravel(),settings):
        x,y,data=curve(rows,name)
        ax.plot(x,y,'o-',label='Round 2 certified candidates',lw=1.8,color='#1c638b')
        x1,y1,_=curve(old,name)
        ax.plot(x1,y1,'x--',label='Round 1 candidates',color='#aaa',lw=1.2)
        if any(r['published_reference_L2'] for r in data):
            ref=np.array([float(r['published_reference_L2']) for r in data])
            ax.plot(x,ref,'s--',label='Clipped-CHSH reference' if name.startswith('fritz') else '2020 Eq. (4) estimate',color='#c65d31',lw=1.2,markersize=4)
        if threshold is not None:
            label = f'CHSH boundary = {threshold:.3g}' if name.startswith('fritz') else f'2020 estimated transition ≈ {threshold:.3g}'
            ax.axvline(threshold,color='#777',ls=':',label=label)
        ax.set_title(title);ax.set_xlabel(r'$u^2$' if name=='renou-u2' else 'Visibility / detector efficiency')
        ax.set_ylabel('Euclidean distance');ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Finite triangle-local search: round 2 benchmark',fontsize=15)
    fig.supxlabel('Equation (4) curves are approximate references, not digitized neural-network data. Residuals do not prove nonlocality.',fontsize=9)
    fig.savefig(out/'noise-curves.png',dpi=160);fig.savefig(out/'noise-curves.svg');plt.close(fig)
    print('Wrote benchmarks/noise-curves.png and noise-curves.svg')


if __name__=='__main__':main()
