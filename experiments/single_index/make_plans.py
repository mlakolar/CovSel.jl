"""Generate prespecified plans; this script does no numerical simulation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
base=dict(n=600,p=40,d=10,index_sparsity=2,graph_degree=4,design="gaussian",graph="block",varying=False,
          curvature=1.0,signal=0.7,noise="gaussian",grid_min=-0.75,grid_max=0.75,grid_n=11,
          h_factors=[1.0],pilot_factors=[1.0],penalties=[0.7],index_modes=["known"],
          methods=["oracle","sqrtlasso","scaled_block","scaled_onestep","aclime_block","aclime_onestep"],
          draws=399,norms=True,replicates=200)
def write(name, scenarios, seed, shard=10):
    lines=[f'seed_base = {seed}',f'shard_size = {shard}']
    for i,c in enumerate(scenarios,1):
        c=dict(base,**c,id=i)
        lines+=['','[[scenario]]']
        lines += [f'{key} = {json.dumps(value)}' for key,value in c.items()]
    (ROOT/(name+'.toml')).write_text('\n'.join(lines)+'\n')
write('smoke',[dict(name='smoke',n=160,p=10,grid_n=3,replicates=2,index_modes=['known','estimated'],draws=199)],17000000,2)
write('development',[
    dict(name='final_bandwidth',h_factors=[0.65,1.0,1.4]),
    dict(name='pilot_bandwidth',pilot_factors=[0.65,1.0,1.6]),
    dict(name='penalty',penalties=[0.35,0.7,1.2]),
    dict(name='varying_nuisance',varying=True,pilot_factors=[0.65,1.0,1.6]),
    dict(name='generated_index',index_modes=['known','estimated'],methods=['oracle','sqrtlasso']),
    dict(name='data_driven',select_tuning=True,methods=['sqrtlasso'],replicates=100),
],23000000)
# A separate seed namespace prevents tuning and confirmation from sharing data.
write('confirmation',[
    dict(name='block_n600'),dict(name='block_n1200',n=1200),
    dict(name='high_dimensional',n=400,p=160),dict(name='varying',varying=True),
    dict(name='chain',graph='chain'),dict(name='grid',graph='grid',p=25),
    dict(name='t8',noise='t8'),dict(name='mixture',noise='mixture'),
    dict(name='curvature',curvature=2.0),
    dict(name='bump_index',design='bump',index_modes=['known','estimated'],methods=['oracle','sqrtlasso']),
    dict(name='sparse_index_d100',d=100,index_sparsity=4,index_modes=['known','estimated'],methods=['oracle','sqrtlasso']),
    dict(name='selected',select_tuning=True,methods=['sqrtlasso']),
],51000000)
write('paper',[
    dict(name=f'{graph}_n{n}_p{p}',n=n,p=p,graph=graph,replicates=1000,draws=1999,grid_n=41)
    for graph in ['block','chain','grid'] for n,p in [(400,80),(800,160),(1600,320)]
]+[
    dict(name='varying_pilot_interaction',varying=True,n=1000,p=80,h_factors=[0.65,1.0,1.4],pilot_factors=[0.65,1.0,1.6],replicates=500,draws=999,grid_n=41),
    dict(name='feasible_index',n=1200,p=80,d=100,index_sparsity=4,index_modes=['known','estimated'],replicates=1000,draws=1999,grid_n=41,methods=['oracle','sqrtlasso','scaled_block','aclime_block']),
    dict(name='selected_independent',select_tuning=True,n=1000,p=80,replicates=1000,draws=1999,grid_n=41,methods=['sqrtlasso']),
    dict(name='heavy_tail',noise='t8',n=1000,p=80,replicates=1000,draws=1999,grid_n=41),
    dict(name='contamination',noise='mixture',n=1000,p=80,replicates=1000,draws=1999,grid_n=41),
    dict(name='boundary',design='uniform',grid_min=-1.0,grid_max=1.0,n=1000,p=80,replicates=2000,draws=1999,grid_n=41),
],91000000,10)

# Refined after the independent 30-replication scout. These fresh seeds evaluate
# support refits and a corrected CV criterion with fixed validation windows.
refit_methods=["oracle","sqrtlasso","sqrt_refit","scaled_block","scaled_refit_block",
               "scaled_onestep","scaled_refit_onestep","aclime_block","aclime_refit_block",
               "aclime_onestep","aclime_refit_onestep"]
write('refinement',[
    dict(name='refit_baseline',methods=refit_methods,replicates=300,norm_refits=[False,True]),
    dict(name='refit_varying',varying=True,methods=refit_methods,replicates=300,norm_refits=[False,True]),
    dict(name='fixed_window_cv',select_tuning=True,methods=['sqrtlasso'],replicates=200),
]+[
    dict(name=f'grid_{g}',grid_n=g,n=1000,seed_group=40,methods=['oracle','sqrtlasso','scaled_refit_block'],replicates=200,draws=999,norms=False)
    for g in [21,41,81]
],187000000)
# Paper-focused holdout after development/refinement; these outcomes do not
# choose the rule being tested. Full broad paper.toml remains conditional.
write('paper_confirmation',[
    dict(name='block_wide_pilot',n=600,p=40,pilot_factors=[1.6],penalties=[0.35,0.7],
         methods=['oracle','sqrtlasso','scaled_refit_block','aclime_refit_block'],
         replicates=1000,draws=1999,grid_n=21,norm_refits=[False,True]),
    dict(name='varying_wide_pilot',n=1200,p=80,varying=True,pilot_factors=[1.6],penalties=[0.35,0.7],
         methods=['oracle','sqrtlasso','scaled_refit_block','aclime_refit_block'],
         replicates=1000,draws=1999,grid_n=21,norm_refits=[False,True]),
    dict(name='cv_high_dimensional',n=400,p=160,select_tuning=True,methods=['sqrtlasso'],
         replicates=1000,draws=1999,grid_n=21,norms=False),
    dict(name='cv_generated_index',n=1200,p=80,d=100,index_sparsity=4,select_tuning=True,
         methods=['sqrtlasso'],index_modes=['known','estimated'],
         replicates=1000,draws=1999,grid_n=21,norms=False),
],317000000,10)
