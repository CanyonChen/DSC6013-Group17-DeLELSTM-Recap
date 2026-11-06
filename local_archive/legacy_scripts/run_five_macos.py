import concurrent.futures, subprocess, os, pathlib, shutil, json, time, sys
root=pathlib.Path(__file__).resolve().parent
code=root/'DeLELSTM-main'; dest=root/'results_five_independent'; logs=root/'logs_five_independent'; stage=root/'staging_five_independent'
for d in [dest,logs,stage]: d.mkdir(exist_ok=True)
folders={'Electricity':'Electricity','PM':'newrevision_pmc','Exchange':'Exchange'}
def run(task):
 name,i=task
 for attempt in range(6):
  seed=555+i+(1 if name=='PM' else 0)+attempt*100
  tmp=stage/f'{name}_{i}_{attempt}';tmp.mkdir(exist_ok=True)
  log=logs/f'{name}_run{i}_attempt{attempt}_seed{seed}.log'
  env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DELE_TORCH_SEED=str(seed))
  env.pop('DELE_MODELS',None);env.pop('DELE_EPOCHS',None)
  with log.open('w') as f:
   f.write(f'dataset={name} run={i} attempt={attempt} torch_seed={seed} split_seed=555 epochs=default\n');f.flush()
   p=subprocess.run([sys.executable,'-u',str(code/(name+'.py')),'--seed','555','--num_exp','1','--save_dirs',str(tmp)],cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT)
  src=tmp/folders[name]/'exp_555_0'
  files=list(src.glob('*/*_Explain_test_results.csv')) if src.exists() else []
  if p.returncode==0 and len(files)==5:
   target=dest/folders[name]/f'exp_555_{i}';target.parent.mkdir(exist_ok=True)
   if target.exists(): raise RuntimeError(f'target already exists: {target}')
   shutil.move(str(src),str(target))
   (target/'run_metadata.json').write_text(json.dumps(dict(dataset=name,run=i,attempt=attempt,torch_seed=seed,split_seed=555),indent=2))
   print(f'COMPLETE {name} run={i} seed={seed}',flush=True);return True
  print(f'FAILED {name} run={i} attempt={attempt} seed={seed} code={p.returncode}; log={log}',flush=True)
 return False
jobs=[(d,i) for i in range(5) for d in ('Exchange','PM','Electricity')]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 ok=list(pool.map(run,jobs))
print(f'FINISHED {sum(ok)}/{len(ok)}',flush=True)
sys.exit(0 if all(ok) else 1)
