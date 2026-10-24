"""Summarize saved results under Slurm. Failed fits count as noncoverage."""
import csv, json, math, sys
from pathlib import Path
from collections import defaultdict

def mean(x): return sum(x)/len(x) if x else float('nan')
def finite(x): return [v for v in x if math.isfinite(v)]
def sd(x):
    if len(x)<2:return float('nan')
    m=mean(x)
    return math.sqrt(sum((z-m)**2 for z in x)/(len(x)-1))
def wilson(k,n):
    z=1.959963984540054
    p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-half,center+half

def main(plan,out):
    root=Path(out)
    # Avoid requiring Python 3.11 on Discovery; the saved task count is authoritative
    # for completion and the plan itself is retained as the scientific definition.
    expected=int((root/'task-count.txt').read_text())
    paths=sorted(root.glob('shard-*.csv'))
    if len(paths)!=expected:raise RuntimeError(f'expected {expected} shards, found {len(paths)}')
    groups=defaultdict(list)
    selected={}
    pointwise=[]
    unique=set()
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                identity=tuple(row[k] for k in ['scenario','rep','method','index_mode','h_factor','pilot_factor','penalty','kind','entry_a','entry_b','point','metric'])
                if identity in unique:raise RuntimeError(f'duplicate row: {identity}')
                unique.add(identity)
                if row['kind']=='pointwise':
                    pointwise.append(row)
                    selected[tuple(row[k] for k in ['scenario','rep','index_mode','h_factor','pilot_factor','penalty'])]=tuple(row[k] for k in ['selected_h','selected_pilot_h','selected_penalty'])
                key=tuple(row[k] for k in ['scenario','method','index_mode','h_factor','pilot_factor','penalty','kind','entry_a','entry_b','point','metric'])
                groups[key].append(row)
    if (root/'expected-pointwise-count.txt').exists():
        expected_rows=int((root/'expected-pointwise-count.txt').read_text())
        if len(pointwise)!=expected_rows:raise RuntimeError(f'expected {expected_rows} pointwise rows, found {len(pointwise)}')
    with (root/'selected_tuning.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['scenario','rep','index_mode','h_factor','pilot_factor','penalty','selected_h','selected_pilot_h','selected_penalty'])
        for key,value in sorted(selected.items()):w.writerow(list(key)+list(value))
    # Oracle comparisons share the realized dataset and exactly the same final h.
    oracles={tuple(r[k] for k in ['scenario','rep','h_factor','pilot_factor','penalty','entry_a','entry_b','point']):r
             for r in pointwise if r['method']=='oracle' and r['index_mode']=='known'}
    with (root/'oracle_gap.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['scenario','rep','method','index_mode','entry_a','entry_b','point','gap','gap_over_oracle_se'])
        for r in pointwise:
            key=tuple(r[k] for k in ['scenario','rep','h_factor','pilot_factor','penalty','entry_a','entry_b','point'])
            o=oracles.get(key)
            if not o or r['failure'] or o['failure'] or r['selected_h']!=o['selected_h']:continue
            gap=float(r['estimate'])-float(o['estimate'])
            w.writerow([r['scenario'],r['rep'],r['method'],r['index_mode'],r['entry_a'],r['entry_b'],r['point'],gap,gap/float(o['se'])])
    columns=['scenario' ,'method','index_mode','h_factor','pilot_factor','penalty','kind','entry_a','entry_b','point','metric','replications','failures','coverage','mcse','wilson_lower','wilson_upper','conditional_coverage','mean_length','bias','empirical_sd','rms_se','mean_metric']
    output=[]
    for key,rows in sorted(groups.items()):
        n=len(rows); valid=[r for r in rows if not r['failure']]
        k=sum(int(r['covered']) for r in rows)
        coverage=k/n
        estimates=finite([float(r['estimate']) for r in valid])
        errors=finite([float(r['estimate'])-float(r['truth']) for r in valid])
        ses=finite([float(r['se']) for r in valid])
        lo,hi=wilson(k,n)
        infer=key[6]!='norm'
        output.append(list(key)+[n,n-len(valid),coverage if infer else '',math.sqrt(coverage*(1-coverage)/n) if infer else '',lo if infer else '',hi if infer else '',k/len(valid) if infer and valid else '',mean(finite([float(r['length']) for r in valid])),mean(errors),sd(estimates),math.sqrt(mean([x*x for x in ses])),mean(finite([float(r['value']) for r in valid]))])
    with (root/'summary.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(columns);w.writerows(output)
    # Paired coverage differences: align shared datasets across methods, including failures.
    paired=defaultdict(lambda:defaultdict(dict))
    for key,rows in groups.items():
        if key[6]=='norm':continue
        comparison=(key[0],)+key[2:]
        for row in rows:paired[comparison][key[1]][row['rep']]=int(row['covered'])
    with (root/'paired.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['comparison','method','reference','n','coverage_difference','mcse'])
        for key,methods in sorted(paired.items()):
            reference='oracle' if 'oracle' in methods else sorted(methods)[0]
            for method,values in sorted(methods.items()):
                if method==reference:continue
                ids=sorted(set(values)&set(methods[reference]))
                d=[values[i]-methods[reference][i] for i in ids]
                w.writerow([json.dumps(key),method,reference,len(d),mean(d),sd(d)/math.sqrt(len(d))])
    receipt={'status':'outputs_present_and_summarized','plan':str(plan),'shards':len(paths),'expected_shards':expected,'rows':len(unique),'summary_groups':len(groups),'checks':['expected shard count','no duplicate result identities','failed fits retained as noncoverage','Wilson intervals and Monte Carlo standard errors','paired method comparisons on common seeds'],'limitations':['No independent scientific refit is implied by these aggregation checks.','Finite-grid bands do not certify continuum coverage.']}
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main(*sys.argv[1:])
