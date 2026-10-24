"""Scientific reporting from completed saved outputs; run under Slurm."""
import csv, math, sys
from collections import defaultdict
from pathlib import Path

def rows(path):
    with Path(path).open() as stream:yield from csv.DictReader(stream)
def fmt(x):
    try:return f'{float(x):.3f}'
    except (ValueError,TypeError):return str(x)
def mean(v):return sum(v)/len(v) if v else float('nan')
def sd(v):return math.sqrt(sum((x-mean(v))**2 for x in v)/(len(v)-1)) if len(v)>1 else float('nan')
def table(lines,header,data):
    lines+=['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in data]
    lines+=['']

def main(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    cases={name:root/('20261024-sig-'+name)/'results' for name in ['development','confirmation','refinement']}
    data={name:list(rows(path/'summary.csv')) for name,path in cases.items()}
    lines=['# Coverage and pilot assessment','','Nominal level: 95%. Failures count as noncoverage. The studies use independent seed namespaces. Widths and bias are conditional on a valid fit. Wilson intervals describe Monte Carlo uncertainty; they do not cover untested designs.','']
    lines+=['## Independent fixed-baseline confirmation','', 'Each cell uses 200 replications. The penalty multiplier is 0.7; final and pilot bandwidths are both 2.5*n^(-1/5). The index is known except where explicitly labeled.','']
    selected=[]
    for r in data['confirmation']:
        if r['kind']=='family':
            selected.append([r['scenario'],r['method'],r['index_mode'],r['replications'],r['failures'],fmt(r['coverage']),f"[{fmt(r['wilson_lower'])}, {fmt(r['wilson_upper'])}]",fmt(r['mean_length'])])
    table(lines,['Scenario','Method','Index','NMC','Failures','Family coverage','Wilson 95% CI','Mean width'],selected)
    lines+=['## Bandwidth and penalty sensitivity','','The development results compare settings on the same simulated datasets. They are calibration evidence, not independent validation of a selected constant.','']
    selected=[]
    for r in data['development']:
        if r['kind']=='family' and r['scenario'] in ('final_bandwidth','pilot_bandwidth','penalty','varying_nuisance'):
            selected.append([r['scenario'],r['method'],r['h_factor'],r['pilot_factor'],r['penalty'],r['replications'],fmt(r['coverage']),fmt(r['mean_length']),r['failures']])
    table(lines,['Scenario','Method','h factor','Pilot factor','Penalty','NMC','Family coverage','Mean width','Failures'],selected)
    lines+=['## Support-refit refinement','','These fresh data evaluate changes motivated by the earlier scout. The fixed bandwidths and penalty were not selected using these outcomes.','']
    selected=[]
    for r in data['refinement']:
        if r['scenario'] in ('refit_baseline','refit_varying') and (r['kind']=='family' or (r['kind']=='pointwise' and abs(float(r['point']))<1e-10)):
            target='joint family' if r['kind']=='family' else f"({r['entry_a']},{r['entry_b']}) at 0"
            selected.append([r['scenario'],r['method'],target,r['replications'],fmt(r['coverage']),f"[{fmt(r['wilson_lower'])}, {fmt(r['wilson_upper'])}]",fmt(r['mean_length']),fmt(r['bias']),r['failures']])
    table(lines,['Scenario','Method','Target','NMC','Coverage','Wilson 95% CI','Mean width','Bias','Failures'],selected)
    lines+=['## Data-driven tuning','','The initial selector scored different candidates on different validation windows. The final fixed-window implementation corrects that comparison. These are separate seed namespaces; the before/after comparison is not paired. A prediction-based criterion need not optimize coverage error.','']
    selected=[]
    for name,scenario in [('development','data_driven'),('confirmation','selected'),('refinement','fixed_window_cv')]:
        for r in data[name]:
            if r['scenario']==scenario and (r['kind']=='family' or (r['kind']=='pointwise' and abs(float(r['point']))<1e-10)):
                target='joint family' if r['kind']=='family' else f"({r['entry_a']},{r['entry_b']}) at 0"
                selected.append([name,scenario,target,r['replications'],fmt(r['coverage']),f"[{fmt(r['wilson_lower'])}, {fmt(r['wilson_upper'])}]",fmt(r['mean_length']),r['failures']])
    table(lines,['Study','Selector','Target','NMC','Coverage','Wilson 95% CI','Mean width','Failures'],selected)
    lines+=['## Full-matrix errors at zero','','The full one-step matrix need not be positive definite. The matrix-norm task differs from inference on a few prespecified entries. These are mean errors, with failed matrices excluded and their counts shown.','']
    selected=[]
    for name,scenario in [('confirmation','block_n600'),('confirmation','high_dimensional'),('refinement','refit_baseline')]:
        for r in data[name]:
            if r['scenario']==scenario and r['kind']=='norm' and r['metric'] in ('max_entry','frobenius','spectral','induced_l1','induced_linf'):
                selected.append([scenario,r['method'],r['metric'],r['replications'],fmt(r['mean_metric']),r['failures']])
    table(lines,['Scenario','Method','Norm','NMC','Mean error','Failures'],selected)
    lines+=['## Nested-grid multiplier diagnostics','','The three nested grids share each dataset and its observation multipliers. Monotonicity of the simulated critical value is checked within replication. Stability is a numerical diagnostic, not a continuum-coverage proof.','']
    records=defaultdict(dict)
    for path in sorted(cases['refinement'].glob('shard-*.csv')):
        for r in rows(path):
            if r['kind']=='family' and r['scenario'].startswith('grid_'):
                records[(r['method'],r['rep'])][int(r['scenario'].split('_')[1])]=r
    griddata=defaultdict(list)
    for (method,rep),values in records.items():
        if len(values)!=3:raise RuntimeError('incomplete grid family')
        a,b,c=(values[g] for g in (21,41,81))
        valid=not any(r['failure'] for r in (a,b,c))
        monotone=valid and float(a['value']) <= float(b['value'])+1e-10 and float(b['value']) <= float(c['value'])+1e-10
        griddata[method].append((valid,monotone,float(c['value'])-float(b['value']) if valid else float('nan'),int(c['covered'])-int(b['covered'])))
    selected=[]
    for method,values in sorted(griddata.items()):
        diffs=[x[3] for x in values]
        selected.append([method,len(values),sum(not x[0] for x in values),sum(x[0] and not x[1] for x in values),fmt(mean([x[2] for x in values if x[0]])),fmt(mean(diffs)),fmt(sd(diffs)/math.sqrt(len(diffs)))])
    table(lines,['Method','Paired NMC','Failed triples','Critical monotonicity violations','Mean c81-c41','Coverage81-Coverage41','Paired MCSE'],selected)
    (out/'ASSESSMENT.md').write_text('\n'.join(lines).rstrip()+'\n')
    # A compact machine-readable table drives the human summary and figures.
    with (out/'headline.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=['study']+list(data['confirmation'][0]),lineterminator='\n')
        w.writeheader()
        for name,rr in data.items():
            for r in rr:
                if r['kind']=='family' or (r['kind']=='pointwise' and abs(float(r['point']))<1e-10):w.writerow(dict(study=name,**r))
    print('Assessment written from three completed studies; no scientific refits performed.')
if __name__=='__main__':main(*sys.argv[1:])
