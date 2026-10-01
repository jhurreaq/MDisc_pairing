#!/usr/bin/env python3
"""Run cases twice in isolation and compare all result/log/PDF bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all','synthetic','treloar','cardiac'),default='all')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workdir',type=Path)
    parser.add_argument('--quick',action='store_true',help='Reduced grids and one noise level; not the full validation')
    args=parser.parse_args()
    directory=args.workdir or Path(tempfile.mkdtemp(prefix='mdisc-validation-'))
    directory.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    environment=dict(os.environ,PYTHONPATH=str(source_root),MPLBACKEND='Agg',
                     MPLCONFIGDIR=str(directory/'matplotlib-cache'),PYTHONHASHSEED='0',
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SOURCE_DATE_EPOCH='0')
    for run in ('first','second'):
        output=directory/run
        if output.exists():
            parser.error(f'Output already exists: {output}; choose a fresh workdir')
        command=[sys.executable,'-m','mdisc',args.case,'--project-root',str(args.project_root),
                 '--output-dir',str(output)]
        if args.quick:
            command += ['--folds','3','--alpha-count','8','--path-count','5','--noise-levels','0','--points','12','--models','MR2']
        print(f'Running {run}: {args.case}',flush=True)
        with (directory/f'{run}.stdout').open('w') as stdout, (directory/f'{run}.stderr').open('w') as stderr:
            subprocess.run(command,env=environment,cwd=directory,stdout=stdout,stderr=stderr,check=True)
    first,second=hashes(directory/'first'),hashes(directory/'second')
    changed=sorted(name for name in first.keys()|second.keys() if first.get(name)!=second.get(name))
    report={'case':args.case,'quick':args.quick,'matching':not changed,'files':len(first),
            'pdfs':sum(name.endswith('.pdf') for name in first),'differing_files':changed,'sha256':first}
    (directory/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))
    return int(bool(changed))

if __name__=='__main__':
    raise SystemExit(main())
