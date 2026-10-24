"""Report an independent 1000-replication study from saved outputs under Slurm."""
import csv,sys
from collections import Counter,defaultdict
from pathlib import Path

def fmt(x):
    try:return f'{float(x):.3f}'
    except (ValueError,TypeError):return str(x)
def table(lines,header,data):
    lines+=['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in data]
    lines+=['']

def main(out):
    out=Path(out)
    rr=list(csv.DictReader((out/'summary.csv').open()))
    lines=['# Independent paper-focused confirmation','','Each configuration uses 1000 replications, 1999 Gaussian observation-multiplier draws and a 21-point index grid. Failures count as noncoverage. The calibration data and this study use different seeds. The rules and comparison constants were fixed before these outcomes.','',
           'The two fixed-rule scenarios use pilot bandwidth 1.6 times the final bandwidth and prespecified penalty multipliers 0.35 and 0.7. The CV scenarios use the final fixed-window selector; their nominal-factor column labels the requested configuration, while the actual selected multipliers are reported in the selection-frequency table. They assess procedures on the supplied index; the estimated-index mode also estimates that index from the same data.','']
    data=[]
    for r in rr:
        if r['kind']=='family':data.append([r['scenario'],r['method'],r['index_mode'],r['penalty'],r['replications'],r['failures'],fmt(r['coverage']),f"[{fmt(r['wilson_lower'])}, {fmt(r['wilson_upper'])}]",fmt(r['mean_length'])])
    table(lines,['Scenario','Method','Index','Nominal factor','NMC','Failures','Joint coverage','Wilson 95% CI','Mean width'],data)
    lines+=['## Pointwise coverage at zero','']
    data=[]
    for r in rr:
        if r['kind']=='pointwise' and abs(float(r['point']))<1e-10:
            data.append([r['scenario'],r['method'],r['index_mode'],r['penalty'],f"({r['entry_a']},{r['entry_b']})",fmt(r['coverage']),f"[{fmt(r['wilson_lower'])}, {fmt(r['wilson_upper'])}]",fmt(r['bias']),fmt(r['empirical_sd']),fmt(r['rms_se']),fmt(r['mean_length']),r['failures']])
    table(lines,['Scenario','Method','Index','Nominal factor','Entry','Coverage','Wilson 95% CI','Bias','Empirical SD','RMS SE','Mean length','Failures'],data)
    lines+=['## Curvewise coverage by entry','']
    data=[]
    for r in rr:
        if r['kind']=='curve':data.append([r['scenario'],r['method'],r['index_mode'],r['penalty'],f"({r['entry_a']},{r['entry_b']})",fmt(r['coverage']),fmt(r['mean_length']),r['failures']])
    table(lines,['Scenario','Method','Index','Nominal factor','Entry','Curve coverage','Mean width','Failures'],data)
    lines+=['## Data-driven selections','','These are observed selection frequencies, not coverage-optimality claims.','']
    selections=defaultdict(Counter)
    for r in csv.DictReader((out/'selected_tuning.csv').open()):
        if r['scenario'].startswith('cv_'):
            selections[(r['scenario'],r['index_mode'])][tuple(r[k] for k in ['selected_h','selected_pilot_h','selected_penalty'])]+=1
    data=[]
    for (scenario,mode),counts in sorted(selections.items()):
        for settings,n in counts.most_common(8):data.append([scenario,mode,*map(fmt,settings),n,fmt(n/sum(counts.values()))])
    table(lines,['Scenario','Index','Final h','Pilot h','Penalty multiplier','Count','Fraction'],data)
    lines+=['## Matrix errors at zero','', 'These are full-matrix errors, not errors of an assembled set of pairwise intervals. Failed matrices are counted separately.','']
    data=[]
    for r in rr:
        if r['kind']=='norm' and r['metric'] in ('max_entry','frobenius','spectral','induced_l1'):
            data.append([r['scenario'],r['method'],r['penalty'],r['metric'],r['replications'],fmt(r['mean_metric']),r['failures']])
    table(lines,['Scenario','Method','Nominal factor','Norm','NMC','Mean error','Failures'],data)
    (out/'PAPER_CONFIRMATION.md').write_text('\n'.join(lines).rstrip()+'\n')
    print('Paper-focused confirmation report written.')
if __name__=='__main__':main(sys.argv[1])
