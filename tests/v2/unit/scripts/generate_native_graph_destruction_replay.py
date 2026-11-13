#!/usr/bin/env python3
"""Compile exact original graph reset bodies with fake SDK return codes; load no runtime."""
from pathlib import Path
import argparse,hashlib
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=args.source;out=args.output;bodies=[]
for cls,rel in [('CUDAGraphCapture','src/v2/backends/cuda/CUDAGraphCapture.cu'),('HIPGraphCapture','src/v2/backends/rocm/HIPGraphCapture.cpp')]:
 s=(source/rel).read_text();a=s.index('    void '+cls+'::reset()');i=s.index('{',a)+1;depth=1
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 body=s[a:i];bodies.append('// Actual production body SHA256 '+hashlib.sha256(body.encode()).hexdigest()+'\n'+body)
out.mkdir(parents=True,exist_ok=True);(out/'NativeGraphDestructionBodies.inc').write_text('\n'.join(bodies)+'\n')
