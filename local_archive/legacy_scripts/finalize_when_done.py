import os, time, subprocess, pathlib, sys
root=pathlib.Path(__file__).resolve().parent
pid=int(sys.argv[1])
while True:
 try: os.kill(pid,0)
 except ProcessLookupError: break
 time.sleep(30)
log=root/'logs_five_macos/supervisor.log'
if 'FINISHED 15/15' not in log.read_text():
 print('Training incomplete; report generation skipped.',flush=True);sys.exit(1)
python='/Users/edz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
for name in ['build_course_paper.py','build_chinese_explanation.py']:
 subprocess.run([python,str(root/name)],check=True,cwd=root)
print('Report generated; visual inspection remains required.',flush=True)
