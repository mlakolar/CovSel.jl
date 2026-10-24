"""Render saved Monte Carlo summaries and targeted tables under Slurm."""
import csv, math, sys
from pathlib import Path

def main(out):
    root=Path(out)
    rows=list(csv.DictReader((root/'summary.csv').open()))
    chosen=[r for r in rows if (r['kind']=='family' or (r['kind']=='pointwise' and abs(float(r['point']))<1e-9))]
    text=['# Simulation summary','','Failures count as noncoverage. Widths and bias use valid fits only.','','| Scenario | Method | Index | h factor | pilot factor | penalty | target | nMC | failures | coverage | MCSE | width | bias | SD / RMS SE |','|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(x):
        try:return f'{float(x):.3f}'
        except (ValueError,TypeError):return x
    for r in chosen:
        target='family' if r['kind']=='family' else f"({r['entry_a']},{r['entry_b']}) at 0"
        ratio=float(r['empirical_sd'])/float(r['rms_se']) if r['kind']=='pointwise' else float('nan')
        text.append('| '+' | '.join([r['scenario'],r['method'],r['index_mode'],r['h_factor'],r['pilot_factor'],r['penalty'],target,r['replications'],r['failures'],f(r['coverage']),f(r['mcse']),f(r['mean_length']),f(r['bias']),f(str(ratio))])+' |')
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    # Standard plotting library. Failure to import is reported without losing results.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        (root/'plot-status.txt').write_text(str(e)+'\n')
        return
    methods=sorted({r['method'] for r in rows if r['kind']=='pointwise'})
    scenarios=sorted({r['scenario'] for r in rows})
    for scenario in scenarios:
        subset=[r for r in rows if r['scenario']==scenario and r['kind']=='pointwise' and r['entry_a']=='1' and r['entry_b']=='1' and r['index_mode']=='known']
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in methods:
            data=[r for r in subset if r['method']==method and r['h_factor']=='1.0' and r['pilot_factor']=='1.0' and r['penalty']=='0.7']
            data.sort(key=lambda r:float(r['point']))
            if not data:continue
            x=[float(r['point']) for r in data];y=[float(r['coverage']) for r in data]
            se=[1.96*float(r['mcse']) for r in data]
            ax.errorbar(x,y,yerr=se,label=method,marker='o',markersize=3,capsize=2)
        ax.axhline(.95,color='black',linestyle='--',linewidth=1)
        ax.set(xlabel='Index value',ylabel='Diagonal pointwise coverage',ylim=(0.5,1.01),title=scenario)
        ax.legend(fontsize=8,ncol=2)
        fig.tight_layout();fig.savefig(root/(scenario+'-diagonal-coverage.pdf'));plt.close(fig)
    (root/'plot-status.txt').write_text('Figures rendered from summary.csv with matplotlib.\n')
if __name__=='__main__':main(sys.argv[1])
