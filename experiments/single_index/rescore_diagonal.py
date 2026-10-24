"""Evaluate log-diagonal endpoints from saved fits; run only under Slurm."""
import csv, math, shutil, sys
from pathlib import Path


def endpoints(row, critical, transform):
    value, se = float(row['estimate']),float(row['se'])
    if transform and row['entry_a']==row['entry_b']:
        if not value>0:
            return float('nan'),float('nan')
        radius=critical*se/value
        return value*math.exp(-radius),value*math.exp(radius)
    return value-critical*se,value+critical*se


def main(source, destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(exist_ok=False)
    for name in ['task-count.txt','expected-pointwise-count.txt','scientific-inputs.sha256']:
        shutil.copyfile(source/name,destination/name)
    for path in sorted(source.glob('shard-*.csv')):
        rows=list(csv.DictReader(path.open()))
        fields=list(rows[0])
        pointwise={}
        for r in rows:
            if r['kind']=='pointwise':
                key=tuple(r[k] for k in ['scenario','rep','method','index_mode','h_factor','pilot_factor','penalty'])
                pointwise.setdefault(key,[]).append(r)
        with (destination/path.name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
            writer.writeheader()
            for r in rows:
                rr=dict(r,method=r['method']+'_log')
                if r['kind']=='pointwise':
                    lo,hi=endpoints(r,1.959963984540054,True)
                    rr.update(covered=int(not r['failure'] and lo<=float(r['truth'])<=hi),length=hi-lo)
                elif r['kind'] in ['family','family_component','curve']:
                    key=tuple(r[k] for k in ['scenario','rep','method','index_mode','h_factor','pilot_factor','penalty'])
                    pp=pointwise[key]
                    if r['kind']!='family':
                        pp=[p for p in pp if (p['entry_a'],p['entry_b'])==(r['entry_a'],r['entry_b'])]
                    cc=float(r['value'])
                    bounds=[endpoints(p,cc,True) for p in pp]
                    cover=all(not p['failure'] and lo<=float(p['truth'])<=hi for p,(lo,hi) in zip(pp,bounds))
                    rr.update(covered=int(cover),length=sum(hi-lo for lo,hi in bounds)/len(bounds))
                writer.writerow(rr)
    print('Log-diagonal endpoint rescoring completed without refitting.')

if __name__=='__main__':main(*sys.argv[1:])
