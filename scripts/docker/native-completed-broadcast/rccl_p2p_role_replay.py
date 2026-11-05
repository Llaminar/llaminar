"""@file rccl_p2p_role_replay.py
@brief Replay the literal typed primitive guard and original diagnostic assertion.

The compiler extracts the actual production constructor fragment. Native and
sealed policies must accept the same role/thread tuples and keep distinct fatal
behavior on invalid tuples. No GPU runtime or execution authority is created.
"""
from pathlib import Path
import subprocess,json,hashlib,os
root=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT']);r=root/'replay/rccl-role';r.mkdir(parents=True,exist_ok=True);source=root/'tree/rccl';s=(source/'src/device/prims_simple.h').read_text();a=s.index('      if constexpr(EntryPolicy==ncclP2pEntryPolicy::SealedLocal256)');b=s.index('assert(2*(nrecv+nsend) <= nthreads);',a)+len('assert(2*(nrecv+nsend) <= nthreads);');snippet=s[a:b]
cpp='''#include "llaminar_p2p_entry.h"
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
template<ncclP2pEntryPolicy EntryPolicy>void check(int nrecv,int nsend,int nthreads){
'''+snippet+'''\n}
int main(){rlimit limit{0,0};setrlimit(RLIMIT_CORE,&limit);int checked=0;
 for(int recv=0;recv<=2;++recv)for(int send=0;send<=2;++send)for(int threads:{0,1,2,3,4,8,31,32,64,128,256}){
  const bool valid=2*(recv+send)<=threads;
  for(int local=0;local<2;++local){
   const auto pid=fork();assert(pid>=0);if(!pid){freopen("/dev/null","w",stderr);
    if(local)check<ncclP2pEntryPolicy::SealedLocal256>(recv,send,threads);
    else check<ncclP2pEntryPolicy::Native>(recv,send,threads);_exit(0);}
   int status=0;assert(waitpid(pid,&status,0)==pid);
   if(valid)assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);
   else assert(WIFSIGNALED(status)&&WTERMSIG(status)==(local?SIGILL:SIGABRT));++checked;
  }
 }printf("PASS actual role predicate: %d Native/Sealed valid/fatal cases\\n",checked);
}
''';cpp=cpp.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>');p=r/'role-check-replay.cpp';p.write_text(cpp);a=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2','-I'+str(source/'src/include'),str(p),'-o',str(r/'role-check-replay')];v=subprocess.run(a,capture_output=True,text=True);(r/'role-check-compile.log').write_text(v.stdout+v.stderr);assert not v.returncode,v.stderr;v=subprocess.run([str(r/'role-check-replay')],capture_output=True,text=True);(r/'role-check.log').write_text(v.stdout+v.stderr);assert not v.returncode,v.stderr;(r/'role-check-result.json').write_text(json.dumps(dict(exit_code=v.returncode,compile_argv=a,source_sha256=hashlib.sha256(s.encode()).hexdigest(),snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),stdout=v.stdout),indent=2));print(v.stdout)
