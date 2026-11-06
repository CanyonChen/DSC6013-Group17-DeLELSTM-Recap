from docx import Document
from pathlib import Path
import pandas as pd
import math
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
root=Path('/Users/edz/Desktop/6013 project')
repro=root/'DS6013DeLELSTM复现_改进版/repro'
rows=[]
for dataset,folder in [('Electricity','Electricity'),('PM2.5','newrevision_pmc'),('Exchange','Exchange')]:
 for e in range(5):
  for model in ['Delelstm','IMV_full','IMV_tensor','Retain','LSTM']:
   d=repro/'results_five_independent'/folder/f'exp_555_{e}'/model
   f=list(d.glob('*_Explain_test_results.csv'))
   if len(f)!=1: raise RuntimeError(f'missing result {d}')
   q=pd.read_csv(f[0]).iloc[0]; metrics=[float(q.rmse_test),float(q.mae_test),float(q.mape_test)*100]
   if not all(math.isfinite(v) for v in metrics): raise RuntimeError(f'nonfinite metric: {f[0]}')
   logs=list(d.glob('*_Expalin_train_results.csv'))
   expected=350 if dataset=='Electricity' else 300
   if len(logs)!=1 or len(pd.read_csv(logs[0]))!=expected: raise RuntimeError(f'incomplete training: {d}')
   rows.append((dataset,model,e,*metrics))
df=pd.DataFrame(rows,columns=['dataset','model','exp','rmse','mae','mape'])
df.to_csv(repro/'results_five_independent'/'verified_detail.csv',index=False)
g=df.groupby(['dataset','model'])[['rmse','mae','mape']].agg(['mean','std'])
g.to_csv(repro/'results_five_independent'/'verified_summary.csv')
doc=Document(root/'02_DeLELSTM_Course_Paper.docx')
title=doc.paragraphs[0]; title.style=doc.styles['Title']; title.text='Replication and Audit of DeLELSTM for Time Series Forecasting'
for run in title.runs: run.font.color.rgb=RGBColor(0,0,0)
title_style=doc.styles['Title'].element.get_or_add_pPr()
for border in title_style.findall(qn('w:pBdr')): title_style.remove(border)
doc.tables[2].cell(2,2).text='Implemented; not used in the five-run fixed-split experiment'
doc.tables[2].cell(5,1).text='Remove a redundant, overwritten assignment'
doc.tables[2].cell(7,2).text='Implemented and used on five-run results'
T=doc.tables[3];T.cell(0,2).text='Replication RMSE mean ± SD (n = 5)'
for row in T.rows[1:]:
 ds=row.cells[0].text.strip(); model=row.cells[1].text.strip(); model={'DeLELSTM':'Delelstm','RETAIN':'Retain','IMV-Full':'IMV_full','IMV-Tensor':'IMV_tensor'}.get(model,model)
 if (ds,model) in g.index:
  v=g.loc[(ds,model),'rmse']; row.cells[2].text=f'{v["mean"]:.6f} ± {v["std"]:.6f}' if ds=='Exchange' else f'{v["mean"]:.4f} ± {v["std"]:.4f}'
def val(ds):
 x=g.loc[(ds,'Delelstm'),'rmse']; return f'{x["mean"]:.6f} ± {x["std"]:.6f}' if ds=='Exchange' else f'{x["mean"]:.4f} ± {x["std"]:.4f}'
winners={ds:g.loc[ds]['rmse']['mean'].idxmin() for ds in ['Electricity','PM2.5','Exchange']}
R={
'DeLELSTM introduces':'DeLELSTM separates instantaneous and historical effects in multivariate forecasting through a linear decomposition of recurrent hidden states. We reproduce the released implementation on Electricity, PM2.5 and Exchange, comparing DeLELSTM with four recurrent baselines over five runs per model and dataset. The DeLELSTM test RMSE means and sample standard deviations are '+val('Electricity')+', '+val('PM2.5')+' and '+val('Exchange')+', respectively. We audit preprocessing, evaluation and explanations. A checkpoint-based sensitivity analysis shows that evaluating all test windows changes RMSE from 1.6739 to 1.6475 for Electricity, from 27.1919 to 26.3707 for PM2.5, and from 0.002343 to 0.002318 for Exchange. These are one-checkpoint comparisons, not five-run estimates. Full-data normalization and random window splitting limit claims about future deployment, and one PM2.5 training attempt produced non-finite values. The results provide a predictive replication and identify concrete limits on the stability and interpretation of the released workflow.',
'The contributions are fourfold.':'This study contributes repeated predictive comparisons, recovery of exported explanation coefficients, a code and protocol audit, and a controlled sensitivity analysis of omitted test windows. The latter evaluates the same saved checkpoints with and without the final incomplete batch and isolates the effect of test-set attrition.',
'The formal GPU results':'',
'The compared models are DeLELSTM, LSTM, RETAIN, IMV-Full, and IMV-Tensor. Default hidden sizes':'The compared models are DeLELSTM, LSTM, RETAIN, IMV-Full, and IMV-Tensor. Batch sizes are 64, 32 and 128 for Electricity, PM2.5 and Exchange, respectively. Training uses MSE loss, Adam optimization, best-validation checkpoint selection, and ReduceLROnPlateau scheduling. Electricity uses 350 epochs and PM2.5 and Exchange use 300. The five-run campaign was executed on CPU using Python 3.13 and PyTorch 2.10.0 on this macOS machine. Each dataset uses a fixed seed-555 random split. Each repetition uses an explicitly recorded independent training seed. A prior PM2.5 seed-555 initialization failed with NaN; independent runs and any retry seeds are recorded in the run metadata, with the split kept at 555. The failed log is retained in logs_five_macos. A retry mechanism retains failed attempt logs and replaces an incomplete run with a new initialization; summaries therefore describe successful runs and do not estimate the unconditional failure rate.',
'The delivered results use one run':'The five repetitions share a fixed split and therefore measure variation in training under that split, not uncertainty across future periods or different splits. The original paper states five repetitions but does not specify enough randomization details to establish exact equivalence. The previously supplied single-run GPU checkpoints are used separately for explanation recovery and the full-test sensitivity analysis.',
'These are substantial usability improvements':'These changes improve usability and result traceability rather than the underlying model or its predictive accuracy. The resampling option changes subset membership while subset sizes remain fixed; folder names use the base seed and experiment index. In addition, the three entry scripts permit a separate training seed so that numerical retries can retain the same data split.',
'Table 3 reports the actual single-run':'Table 3 reports the mean and sample standard deviation of five completed test RMSE values for each dataset and model, alongside the published five-run mean. Lower values are better. The full run-level results and MAE and MAPE summaries are supplied with the code.',
'For DeLELSTM, the replicated RMSE differs':'The replicated DeLELSTM mean ± sample standard deviation is '+val('Electricity')+' on Electricity, '+val('PM2.5')+' on PM2.5, and '+val('Exchange')+' on Exchange. The lowest mean RMSE is obtained by '+winners['Electricity']+' on Electricity, '+winners['PM2.5']+' on PM2.5, and '+winners['Exchange']+' on Exchange. These comparisons are descriptive: the paper and this reproduction may differ in software, hardware and randomization, and no paired significance test is available.',
'The complete Electricity DeLELSTM result':'The full tables of MAE and MAPE accompany the run-level data. Predictions and errors are reported on the target scale used by the released code. Exchange values should be interpreted at more precision than the four decimal places in the original published table.',
'The largest discrepancy occurs':'Baseline differences should be interpreted in light of run-to-run variation. Electricity LSTM test RMSE across the five repetitions is '+', '.join(f'{v:.4f}' for v in df[(df.dataset=='Electricity') & (df.model=='LSTM')].sort_values('exp').rmse)+'; its mean is '+f"{g.loc[('Electricity','LSTM'),('rmse','mean')]:.4f}"+' and sample standard deviation is '+f"{g.loc[('Electricity','LSTM'),('rmse','std')]:.4f}"+'. One high-error run drives much of the variation; the original paper likewise reports a large standard deviation of 6.3693. Five runs cannot establish the frequency or cause of such outcomes. The present results cannot isolate the contribution of initialization, implementation version, hardware, or other protocol choices.',
'The unregularized normal-equation implementation motivates':'The unregularized normal-equation implementation motivates a numerical-stability check. Table 5 summarizes absolute values of previously exported coefficients. These statistics do not measure matrix condition number and cannot establish instability on their own. Separately, the seed-555 PM2.5 training attempt encountered NaN at epoch 85 on this CPU environment, a directly observed failure whose precise cause has not been established.',
'Normalization statistics are computed':'Normalization statistics are computed before the random split and therefore include validation and test information. Exchange windows overlap because the preprocessing script constructs rolling 30-day windows, whereas Electricity uses day groups and PM2.5 uses consecutive 24-hour groups. The random assignment in all three datasets also permits training observations later than test observations. In the seed-555 Exchange split, all 757 test windows overlap a training window and 713 have an adjacent-day training window; among the 640 evaluated windows, the counts are 640 and 603. These choices limit conclusions about deployment on a genuinely future period; they do not establish the direction or size of any bias.',
'Validation and test loaders use drop_last':'Validation and test loaders use drop_last = True. Consequently, the evaluated test set excludes 28 of 92 Electricity windows, 24 of 184 PM2.5 windows, and 117 of 757 Exchange windows. We independently re-evaluated the existing seed-555 DeLELSTM checkpoints on the full test sets using the same split and no retraining. The retained-window RMSE reproduced the saved results (1.6739, 27.1919, 0.002343); including all windows changed RMSE to 1.6475, 26.3707, and 0.002318 respectively. In these three single-checkpoint comparisons, omission increased RMSE; the direction is data-dependent. This sensitivity test does not correct the validation selection protocol and is not a five-run comparison.',
'This is a comprehensive single-run replication':'This study reproduces the principal predictive experiment and audits the implementation, but does not reproduce every experiment or establish causal validity of the explanations. Important limitations are:',
'• Only one formal run':'• The five training repetitions share a split and do not assess variability across different temporal partitions.',
'• The improved repeated-run machinery':'• PM2.5 required a documented initialization retry after a non-finite training failure. This observation does not identify its cause.',
'The highest-priority next step':'The highest-priority follow-up is a controlled comparison using chronological splits, train-only normalization, all validation and test windows, and repeated independent seeds, followed by the sufficiency test. Numerical diagnostics should compare matrix condition numbers, reconstruction residuals, and regularized solvers.',
'The replicated DeLELSTM results are numerically close':'The five-run experiments and the supplied run-level records enable a direct comparison with the original performance table. The code audit and full-test sensitivity study show that a reported test score depends on more than model architecture: data preparation, sample retention, checkpoint selection, and numerical behavior also matter. The explanation results are descriptive model attributions, and the incomplete reproduction of case studies warrants further validation.',
'The improved package contains':'The improved package contains released data and code, patches, training logs, model checkpoints, result tables and explanation scripts. The five-run outputs and their logs are in repro/results_five_independent and repro/logs_five_independent. The earlier GPU single-run artifacts used for the sensitivity analysis are in repro/results. The audit_full_test.py script reconstructs the same split and evaluates the original checkpoint with all test windows.',
'The primary predictive evidence is stored':'The five-run prediction evidence is stored under repro/results_five_independent/{dataset}/exp_555_{0..4}/{model}/. Each *_Explain_test_results.csv contains test metrics; verified_detail.csv and verified_summary.csv provide run-level and aggregated values. The separate full-test audit uses the original checkpoints in repro/results and writes logs_five_macos/audit_full_test.log.',
'This paper does not claim five completed repetitions':'This paper does not claim statistical significance, exact reproduction of every published figure, a completed sufficiency test, or proof of an ill-conditioned least-squares system. These claims require further evidence.'}
for p in doc.paragraphs:
 text=p.text
 for k,v in R.items():
  if text.startswith(k):
   p.text=v;break
# Correct the historical abstract-like limitation labels and unsupported blanket overlap claims
for p in doc.paragraphs:
 if 'random splitting of overlapping windows' in p.text: p.text=p.text.replace('random splitting of overlapping windows','random splitting including overlapping Exchange windows')
 if p.text.startswith('cd repro') or p.text.startswith('.\\run_all') or p.text.startswith('python summarize') or p.text.startswith('python interpret'): p.text=''
 if p.text.startswith('A minimal repeated-run workflow'): p.text='The scripts accept --num_exp 5 and --save_dirs to reproduce the fixed-split campaign; the logs record the actual commands and settings.'
for par in doc.paragraphs:
 if par.text.startswith(('4. Reproducibility Engineering Improvements','5. Predictive Results')): par.paragraph_format.page_break_before=True
for ti in [1,2,3]:
 table=doc.tables[ti]
 for ri,row in enumerate(table.rows):
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if ri==0: trpr.append(OxmlElement('w:tblHeader'))
  for cell in row.cells:
   for par in cell.paragraphs:
    par.paragraph_format.space_before=Pt(0);par.paragraph_format.space_after=Pt(0)
    for run in par.runs: run.font.size=Pt(7.5)
out=root/'04_DeLELSTM_Five_Run_Course_Paper.docx'
doc.save(out)
print(out)
