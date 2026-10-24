"""Report diagonal/CV studies from saved outputs; execute under Slurm."""
import csv, json, math, sys
from pathlib import Path
from collections import Counter,defaultdict


def f(x):
    return f'{float(x):.3f}'


def table(lines, header, rows):
    lines += ['| '+' | '.join(header)+' |', '|'+'|'.join(['---']*len(header))+'|']
    lines += ['| '+' | '.join(map(str, row))+' |' for row in rows]
    lines += ['']


def main(out, phase='development', reference=None):
    out = Path(out)
    rows = list(csv.DictReader((out/'summary.csv').open()))
    lines = [f'# Diagonal and CV {phase}', '',
        'Coverage is unconditional over all requested fits. Failures count as noncoverage. '
        'Intervals are nominal 95%. Bands are joint over the three selected entries and the full grid. '
        'Wilson intervals quantify Monte Carlo uncertainty, not inferential uncertainty in a dataset.', '',
        'Method suffixes: `refit` removes coefficient shrinkage; `df` refits the union support and '
        'normalizes residual-maker row variance; `hc3` adjusts final smoothing leverage; `bc` '
        'subtracts second-order inverse curvature. The components are defined in '
        '`docs/single_index/DIAGONAL_TUNING.md`. Actual tuning values are in `selected_tuning.csv`.', '']
    families = [r for r in rows if r['kind']=='family']
    table(lines, ['Scenario','Method','Index','NMC','Failures','Joint coverage','Wilson 95% CI','Mean width'],
        [[r['scenario'],r['method'],r['index_mode'],r['replications'],r['failures'],f(r['coverage']),
          f"[{f(r['wilson_lower'])}, {f(r['wilson_upper'])}]",f(r['mean_length'])] for r in families])
    lines += ['## Pointwise intervals at zero', '']
    point0 = [r for r in rows if r['kind']=='pointwise' and abs(float(r['point']))<1e-10]
    table(lines, ['Scenario','Method','Index','Entry','Coverage','Bias','Empirical SD','RMS SE','SD/SE','Mean length','Failures'],
        [[r['scenario'],r['method'],r['index_mode'],f"({r['entry_a']},{r['entry_b']})",f(r['coverage']),
          f(r['bias']),f(r['empirical_sd']),f(r['rms_se']),f(float(r['empirical_sd'])/float(r['rms_se'])),
          f(r['mean_length']),r['failures']] for r in point0])
    lines += ['## Curvewise bands', '']
    table(lines, ['Scenario','Method','Index','Entry','Coverage','Mean width','Failures'],
        [[r['scenario'],r['method'],r['index_mode'],f"({r['entry_a']},{r['entry_b']})",f(r['coverage']),
          f(r['mean_length']),r['failures']] for r in rows if r['kind']=='curve'])
    lines += ['## Selected tuning', '', 'Frequencies are descriptive. No truth or coverage entered selection.', '']
    selections = defaultdict(Counter)
    for r in csv.DictReader((out/'selected_tuning.csv').open()):
        if r['scenario'].startswith('cv_'):
            selections[(r['scenario'],r['index_mode'])][tuple(r[k] for k in ['selected_h','selected_pilot_h','selected_penalty'])] += 1
    selection_rows = []
    for (scenario,mode),counts in sorted(selections.items()):
        for setting,count in counts.most_common(5):
            selection_rows.append([scenario,mode,*map(f,setting),count,f(count/sum(counts.values()))])
    table(lines,['Scenario','Index','Final h','Pilot h','Penalty multiplier','Count','Fraction'],selection_rows)
    # Same seed groups share observations even across tuning-rule scenarios.
    values = defaultdict(dict)
    paths = sorted(out.glob('shard-*.csv'))
    if reference:
        paths += sorted(Path(reference).glob('shard-*.csv'))
    for path in paths:
        for r in csv.DictReader(path.open()):
            if r['kind']!='family' and not (r['kind']=='pointwise' and r['entry_a']==r['entry_b'] and abs(float(r['point']))<1e-10):
                continue
            group = (r['index_mode'],r['kind'],r['entry_a'],r['entry_b'])
            values[group][(r['scenario'],r['method'],r['seed'])] = (int(r['covered']),float(r['length']))
    paired = []
    for group, vv in values.items():
        configs = sorted({k[:2] for k in vv})
        for scenario, method in configs:
            if '_prediction' in scenario and method=='sqrtlasso':
                continue
            if scenario.startswith('cv_'):
                stem = scenario.rsplit('_',1)[0]
                ref_s, ref_m = stem+'_prediction', 'sqrtlasso'
                if (ref_s,ref_m) not in configs and (scenario,'sqrt_refit_df_hc3_bc') in configs:
                    ref_s,ref_m = scenario,'sqrt_refit_df_hc3_bc'
            else:
                ref_s = scenario
                ref_m = 'sqrt_refit_block' if (scenario,'sqrt_refit_block') in configs else 'scaled_refit_block'
            current = {k[2]:v for k,v in vv.items() if k[:2]==(scenario,method)}
            reference = {k[2]:v for k,v in vv.items() if k[:2]==(ref_s,ref_m)}
            seeds = sorted(set(current)&set(reference))
            if not seeds or (scenario,method)==(ref_s,ref_m):
                continue
            ds = [current[s][0]-reference[s][0] for s in seeds]
            m = sum(ds)/len(ds)
            se = math.sqrt(sum((d-m)**2 for d in ds)/(len(ds)*(len(ds)-1))) if len(ds)>1 else float('nan')
            wr = sum(current[s][1] for s in seeds)/sum(reference[s][1] for s in seeds)
            paired.append([scenario,method,ref_s,ref_m,*group,len(seeds),m,se,wr])
    with (out/'paired_rules.csv').open('w') as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['scenario','method','reference_scenario','reference_method','index','kind','entry_a','entry_b','n','coverage_difference','mcse','mean_width_ratio'])
        writer.writerows(paired)
    lines += ['## Paired rule comparisons', '',
        'The seed identifies the same generated dataset. The comparison includes support selection, '
        'bandwidth selection, and index estimation. Width ratios use the same replications.', '']
    table(lines, ['Scenario','Method','Reference','Index','Kind','NMC','Coverage difference','MCSE','Width ratio'],
        [[r[0],r[1],r[2]+'/'+r[3],r[4],r[5],r[8],f(r[9]),f(r[10]),f(r[11])] for r in paired])
    (out/'DIAGONAL_TUNING.md').write_text('\n'.join(lines).rstrip()+'\n')
    plot_available = False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_available = True
        scenarios = sorted({(r['scenario'],r['index_mode']) for r in families})
        fig, axes = plt.subplots(len(scenarios),1,figsize=(10,2.4*len(scenarios)),squeeze=False,layout='constrained')
        for ax, (scenario,mode) in zip(axes[:,0],scenarios):
            rr = [r for r in families if (r['scenario'],r['index_mode'])==(scenario,mode)]
            yy = [float(r['coverage']) for r in rr]
            ax.errorbar(range(len(rr)),yy,yerr=[[float(r['coverage'])-float(r['wilson_lower']) for r in rr],
                [float(r['wilson_upper'])-float(r['coverage']) for r in rr]],fmt='o',capsize=3)
            ax.axhline(.95,color='black',linestyle='--',linewidth=.8)
            ax.set_xticks(range(len(rr)),[r['method'] for r in rr],rotation=25,ha='right',fontsize=8)
            ax.set_ylim(min(.8,min(yy)-.03),1.005)
            ax.set_title(scenario+' / '+mode)
            ax.set_ylabel('Joint coverage')
        fig.savefig(out/'diagonal_tuning.png',dpi=140)
        fig.savefig(out/'diagonal_tuning.pdf')
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; table report retained.')
    if phase == 'confirmation' and plot_available:
        # Compact comparison for the two original CV failures and index modes.
        import matplotlib.pyplot as plt
        cases=[('highdim','known'),('generated','known'),('generated','estimated')]
        fig,axes=plt.subplots(2,3,figsize=(10,5),layout='constrained',sharey='row')
        for col,(case,mode) in enumerate(cases):
            for row,kind in enumerate(['pointwise','family']):
                ax=axes[row,col]
                selected=[]
                for rule,method in [('prediction','sqrtlasso'),('inference','sqrt_refit_df_hc3_bc_log')]:
                    matches=[r for r in rows if r['scenario']=='cv_'+case+'_'+rule and
                        r['method']==method and r['index_mode']==mode and r['kind']==kind and
                        (kind!='pointwise' or (r['entry_a']==r['entry_b'] and abs(float(r['point']))<1e-10))]
                    if matches:selected.append(matches[0])
                if len(selected)!=2:continue
                yy=[float(r['coverage']) for r in selected]
                ax.errorbar([0,1],yy,yerr=[[float(r['coverage'])-float(r['wilson_lower']) for r in selected],
                    [float(r['wilson_upper'])-float(r['coverage']) for r in selected]],fmt='o',capsize=4,color='#2455a4')
                ax.axhline(.95,color='black',linestyle='--',linewidth=.8)
                ax.set_xticks([0,1],['Prediction CV','Revised rule'],fontsize=9)
                ax.set_xlim(-.4,1.4)
                ax.set_ylim(.83,1.)
                if col==0:ax.set_ylabel('Diagonal at zero' if kind=='pointwise' else 'Joint grid coverage')
                if row==0:ax.set_title(case+' / '+mode,fontsize=10)
                for x,y in enumerate(yy):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=.6))
        fig.savefig(out/'cv_confirmation.png',dpi=180)
        fig.savefig(out/'cv_confirmation.pdf')
        plt.close(fig)
    for r in families:
        print(r['scenario'],r['method'],r['index_mode'],f(r['coverage']),r['failures'],f(r['mean_length']))
    print('Diagonal/CV report and matched-seed comparisons written.')

if __name__=='__main__':
    main(*sys.argv[1:])
