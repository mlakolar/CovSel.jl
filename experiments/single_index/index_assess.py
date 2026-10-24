"""Report first-stage alignment CV from saved cluster outputs."""
import csv, math, sys
from collections import Counter,defaultdict
from pathlib import Path
from diagonal_assess import table,f


def main(root):
    root=Path(root)
    rows=list(csv.DictReader((root/'summary.csv').open()))
    lines=['# First-stage index tuning', '',
        'The second-stage guarded CV and diagonal correction are fixed. Index modes compare known indices, '
        'the original multiplier 0.5 with soft thresholding, and held-out Stein-alignment CV with optional '
        'support unshrinking. No true index or coverage enters tuning. Failures count as noncoverage.', '']
    families=[r for r in rows if r['kind']=='family']
    table(lines,['Scenario','Method','Index','NMC','Failures','Joint coverage','Wilson 95% CI','Mean width'],
        [[r['scenario'],r['method'],r['index_mode'],r['replications'],r['failures'],f(r['coverage']),
          f"[{f(r['wilson_lower'])}, {f(r['wilson_upper'])}]",f(r['mean_length'])] for r in families])
    point0=[r for r in rows if r['kind']=='pointwise' and abs(float(r['point']))<1e-10]
    lines+=['## Pointwise intervals and index error', '',
        'Index error means are conditional on a successful fit. Index failures remain in coverage denominators.', '']
    table(lines,['Scenario','Method','Index','Entry','Coverage','Bias','SD','RMS SE','Mean direction error','Failures'],
        [[r['scenario'],r['method'],r['index_mode'],f"({r['entry_a']},{r['entry_b']})",f(r['coverage']),
          f(r['bias']),f(r['empirical_sd']),f(r['rms_se']),f(r['mean_metric']),r['failures']] for r in point0])
    counts=defaultdict(Counter)
    for r in csv.DictReader((root/'selected_tuning.csv').open()):
        if r['index_mode']=='estimated_cv':
            counts[r['scenario']][(r['selected_index_penalty'],r['index_unshrink'],r['index_support'],r.get('index_blocks','5'))]+=1
    lines+=['## Observed first-stage selections', '']
    data=[]
    for scenario,cc in sorted(counts.items()):
        for (m,u,s,b),n in cc.most_common(8):data.append([scenario,m,u,s,b,n,f(n/sum(cc.values()))])
    table(lines,['Scenario','Multiplier','Unshrink','Support','Blocks','Count','Fraction'],data)
    matches=defaultdict(dict)
    for path in sorted(root.glob('shard-*.csv')):
        for r in csv.DictReader(path.open()):
            if r['kind']!='family' and not (r['kind']=='pointwise' and r['entry_a']==r['entry_b'] and abs(float(r['point']))<1e-10):continue
            key=tuple(r[k] for k in ['scenario','method','kind','entry_a','entry_b'])
            matches[key][(r['seed'],r['index_mode'])]=r
    paired=[]
    for key,v in sorted(matches.items()):
        seeds=sorted({s for s,m in v if m=='estimated_cv'}&{s for s,m in v if m=='estimated'})
        if not seeds:continue
        ds=[int(v[s,'estimated_cv']['covered'])-int(v[s,'estimated']['covered']) for s in seeds]
        mean=sum(ds)/len(ds)
        se=math.sqrt(sum((d-mean)**2 for d in ds)/(len(ds)*(len(ds)-1)))
        paired.append([*key,len(ds),mean,se])
    with (root/'paired_index.csv').open('w') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['scenario','method','kind','entry_a','entry_b','n','coverage_difference','mcse']);w.writerows(paired)
    lines+=['## Paired tuned versus original index comparisons', '']
    table(lines,['Scenario','Method','Kind','NMC','Coverage difference','MCSE'],
        [[r[0],r[1],r[2],r[5],f(r[6]),f(r[7])] for r in paired])
    (root/'INDEX_TUNING.md').write_text('\n'.join(lines).rstrip()+'\n')
    for r in families:print(r['scenario'],r['method'],r['index_mode'],f(r['coverage']),r['failures'],f(r['mean_length']))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        scenarios=sorted({r['scenario'] for r in families})
        fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for ax,scenario in zip(axes.flat,scenarios):
            rr=[next(r for r in families if r['scenario']==scenario and r['method']=='sqrt_refit_df_hc3_bc_log' and r['index_mode']==m)
                for m in ['known','estimated','estimated_cv']]
            yy=[float(r['coverage']) for r in rr]
            ax.errorbar(range(3),yy,yerr=[[float(r['coverage'])-float(r['wilson_lower']) for r in rr],
                [float(r['wilson_upper'])-float(r['coverage']) for r in rr]],fmt='o',capsize=4,color='#2455a4')
            ax.axhline(.95,color='black',linestyle='--',linewidth=.8)
            ax.set_xticks(range(3),['Known index','Fixed MOM','Selected index'],fontsize=9)
            ax.set_xlim(-.4,2.4)
            ax.set_ylim(min(.7,min(float(r['wilson_lower']) for r in rr)-.025),1.005)
            ax.set_ylabel('Joint grid coverage')
            ax.set_title(scenario.removeprefix('cv_').replace('_',' '),fontsize=10)
            for x,y in enumerate(yy):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',pad=.6))
        fig.savefig(root/'index_coverage.png',dpi=180)
        fig.savefig(root/'index_coverage.pdf')
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; tables retained.')
    print('Index tuning report written.')

if __name__=='__main__':main(sys.argv[1])
