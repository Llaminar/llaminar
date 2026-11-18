#!/usr/bin/env python3
"""Export byte-identical pinned Castagna device definitions, without host graph plumbing.

No math, declaration, kernel or helper source is rewritten. Only the complete
leading source interval through its anonymous-namespace close is retained.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
PIN='3483d462715d61f21ed5877836be35a724446359'
SOURCE='ggml/src/ggml-cuda/hc-persist.cu'
SOURCE_SHA256='7f3b7040111c9587fdd4ca062cb209077ed00abaee4e321bdd11c3deda4c0308'
BOUNDARY=b'\n} // namespace\n\nint ggml_cuda_hc_persist_mode()'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();root=args.source_root.resolve();out=args.output_dir.resolve()
 if git(root,'rev-parse','HEAD').decode().strip()!=PIN:raise SystemExit('Castagna comparator checkout is not the selected original pin')
 actual=(root/SOURCE).read_bytes();original=git(root,'show',f'{PIN}:{SOURCE}')
 if actual!=original or hashlib.sha256(actual).hexdigest()!=SOURCE_SHA256:raise SystemExit('Castagna kernel source differs from exact original authority')
 if actual.count(BOUNDARY)!=1:raise SystemExit('Missing/ambiguous complete original device-section boundary')
 cut=actual.index(BOUNDARY)+len(b'\n} // namespace\n');section=actual[:cut]
 if section.count(b'__global__ void hc_persist_kernel(')!=1 or not section.endswith(b'} // namespace\n'):raise SystemExit('Incomplete original kernel section')
 # Header closure is owned by this same pin; modified tracked dependency content
 # must not silently enter a source-identical kernel compilation.
 tracked=git(root,'ls-files','ggml/src/ggml-cuda','ggml/src/ggml-hip','ggml/include','ggml/src/ggml-common.h','ggml/src/ggml-impl.h','ggml/src/ggml-cpu/ggml-cpu-impl.h','LICENSE').decode().splitlines()
 dependencies=[]
 for name in tracked:
  path=root/name
  if not path.is_file():raise SystemExit('Missing pinned comparator dependency: '+name)
  if path.suffix not in ('.h','.hpp','.cuh','.inc') and name not in ('LICENSE',SOURCE):continue
  blob=path.read_bytes();expected=git(root,'show',f'{PIN}:{name}')
  if blob!=expected:raise SystemExit('Modified pinned comparator dependency: '+name)
  dependencies.append({'path':name,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)})
 license_bytes=(root/'LICENSE').read_bytes();out.mkdir(parents=True,exist_ok=True)
 (out/'CastagnaOriginalDeviceSection.inc').write_bytes(section);(out/'CASTAGNA-LICENSE').write_bytes(license_bytes)
 result={'pin':PIN,'source_root':str(root),'source':SOURCE,'source_sha256':SOURCE_SHA256,'source_bytes':len(actual),'export_bytes':len(section),'export_sha256':hashlib.sha256(section).hexdigest(),'export_interval':[0,cut],'byte_identical_original_prefix':True,'kernel_body_modified':False,'host_graph_plumbing_excluded':True,'dependencies':dependencies}
 (out/'castagna-source-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='dependencies'},indent=2))
if __name__=='__main__':main()
