"""Generate the three manuscript figures from checked certificate/benchmark data."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tool'))
from triangle_local.exact import fraction
from triangle_local.io import load_json

COLORS={'triangle-local':'#21618c','LHV-Net-2020':'#bf6516','localmodels':'#608344'}
MARKERS={'triangle-local':'o','LHV-Net-2020':'s','localmodels':'^'}
TARGET_LABELS={'ejm-source-080':'EJM source noise\n$v=0.80$',
 'ejm-detector-086':'EJM detector noise\n$v=0.86$',
 'renou-detector-093':'Renou detector noise\n$u^2=0.85$, $v=0.93$'}


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=220,facecolor='white')
    svg=out/(name+'.svg')
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def selected():
    records=json.loads((ROOT/'benchmarks/target-bounds.json').read_text())['records']
    def row(f,p):
        r=next(r for r in records if r['family']==f and r['parameter']==p)
        bundle=load_json(ROOT/'benchmarks'/r['model_file'])
        # Bind every plotted bound to the matching exact certificate target.
        assert bundle['certificate']['target_sha256']==r['stored_target_sha256']
        return float(r['U_sqrtE_plus_delta_target_upper'])
    exact=load_json(ROOT/'benchmarks/models/000-note-41-144-explicit.json.gz')
    assert exact['certificate']['exact_match']
    return [
      ('Exactly symmetric construction',0),
      ('Renou noiseless, $u^2=0.50$',row('renou-u2','1/2')),
      ('EJM source noise, $v=0.80$',row('elegant-source','4/5')),
      ('EJM detector noise, $v=0.86$',row('elegant-detector','43/50')),
      ('Renou source noise, $v=0.89$',row('renou-source','89/100')),
      ('Renou detector noise, $v=0.93$',row('renou-detector','93/100')),
      ('EJM noiseless',row('elegant-source','1')),
      ('Renou noiseless, $u^2=0.85$',row('renou-u2','17/20'))]


def headline(out):
    data=selected()
    fig=plt.figure(figsize=(8.8,3.9))
    gs=fig.add_gridspec(1,2,width_ratios=[1.05,5],left=.33,right=.98,bottom=.18,top=.79,wspace=.12)
    exact=fig.add_subplot(gs[0]);positive=fig.add_subplot(gs[1],sharey=exact)
    fig.suptitle('Certified local approximations to triangle-network correlations',fontsize=14,y=.97)
    fig.text(.5,.88,'Selected development witnesses · Renou noise rows use $u^2=0.85$',ha='center',fontsize=11)
    for i,(label,value) in enumerate(data):
        if value==0:
            exact.scatter([.15],[i],s=42,color='#21618c',marker='D')
            exact.text(.29,i,'0 — exact',va='center',fontsize=11)
        else:
            positive.scatter([value],[i],s=47,color='#21618c',edgecolor='white',linewidth=.7,zorder=3)
    exact.set(yticks=range(len(data)),yticklabels=[x[0] for x in data],xlim=(0,1.2),ylim=(len(data)-.5,-.5),xticks=[])
    exact.tick_params(axis='y',length=0,pad=8,labelsize=11)
    exact.set_xlabel('Exact equality',fontsize=10.5)
    for spine in exact.spines.values(): spine.set_visible(False)
    positive.set(xscale='log',xlim=(1e-11,.15),xlabel='Certified upper bound $U$ on $L_2$ distance')
    positive.set_xticks([1e-10,1e-8,1e-6,1e-4,1e-2])
    positive.tick_params(axis='y',left=False,labelleft=False)
    positive.spines['left'].set_visible(False)
    positive.grid(axis='x',color='#dce2e6')
    positive.set_axisbelow(True)
    save(fig,out,'headline')


def comparison(out):
    protocol=json.loads((ROOT/'benchmarks/round4/protocol.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(8.5,3.8),sharey=True)
    fig.subplots_adjust(left=.105,right=.985,bottom=.19,top=.62,wspace=.12)
    fig.suptitle('Fitting identical targets from random starts',fontsize=16,y=.98)
    methods=protocol['methods']
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],linestyle='none',label=('LHV-Net-2020\n(small CPU configuration)' if m=='LHV-Net-2020' else m)) for m in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.915),ncol=3,frameon=False,fontsize=11)
    for ax,t in zip(axes,protocol['targets']):
        for mi,method in enumerate(methods):
            for rep in range(protocol['repetitions']):
                path=ROOT/'benchmarks/round4/runs'/f"{t['id']}__{method}__{rep:02d}.json"
                run=json.loads(path.read_text())
                for budget in [10,60]:
                    cp=run['checkpoints'][str(budget)]
                    x=budget+(mi-1)*2.8+(rep-4.5)*.35
                    y=cp['l2']
                    if 'ci95' in cp:
                        lo,hi=cp['ci95']
                        # Intervals are plotted at their endpoints; basic bootstrap
                        # intervals need not contain the point estimate.
                        ax.vlines(x,max(lo,1e-14),hi,color=COLORS[method],linewidth=.7,alpha=.65,zorder=2)
                    ax.scatter(x,y,s=19,marker=MARKERS[method],color=COLORS[method],alpha=.78,linewidth=0,zorder=3)
        ax.axhline(protocol['success_threshold_l2'],color='#777777',linestyle=':',linewidth=.8,zorder=1)
        ax.set(title=TARGET_LABELS[t['id']],xlim=(2,68),xticks=[10,60],xlabel='Wall-clock seconds',yscale='log')
        ax.tick_params(labelsize=10.5)
        ax.grid(axis='y',alpha=.3)
    axes[0].set_ylabel('$L_2$ to the common target')
    fig.text(.5,.015,'Ten runs per method and target · dotted line: declared success threshold',ha='center',fontsize=9)
    save(fig,out,'baseline-time')


def compact(out):
    receipt=json.loads((ROOT/'benchmarks/round4/compact-verification.json').read_text())
    bound=next(r for r in json.loads((ROOT/'benchmarks/target-bounds.json').read_text())['records']
               if r['family']=='renou-detector' and r['parameter']=='93/100')
    fig,ax=plt.subplots(figsize=(8.4,2.5));ax.axis('off')
    fig.suptitle('A compact triangle-local witness',fontsize=16,y=.96)
    ax.text(.5,.88,'Renou detector noise · $u^2=0.85$, $v=0.93$',ha='center',fontsize=11,transform=ax.transAxes)
    rows=[['Source sizes','$(12,12,12)\\;\\rightarrow\\;(8,9,2)$'],
          ['Bound to intended target','$U\\leq 8.208\\times10^{-9}$'],
          ['Compressed witness',f"{receipt['bundle_bytes']:,} bytes"],
          ['Exact checking',f"{1000*receipt['median_check_seconds']:.2f} ms"]]
    tab=ax.table(cellText=rows,colWidths=[.45,.52],cellLoc='left',bbox=[.06,.08,.88,.70])
    tab.auto_set_font_size(False);tab.set_fontsize(11)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#d8e0e6');cell.set_facecolor('#f4f7fa' if r%2==0 else 'white')
    fig.text(.5,.035,'Exact removal of zero-weight symbols; no minimum-cardinality claim',ha='center',fontsize=9)
    save(fig,out,'local-model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'docs/figures')
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'triangle-local-round4'})
    headline(args.out_dir);comparison(args.out_dir);compact(args.out_dir)
    print('Wrote and data-checked headline, three-method comparison, and compact witness box.')


if __name__=='__main__': main()
