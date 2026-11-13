#!/usr/bin/env python3
## @file retirement_replay.py
# @brief Replay native proxy cancellation and last-owner storage retirement.
"""Replay the literal native proxy loop predicate and existing retirement store.

Host abort must still wake GROUP jobs; closing those peers cannot release SHM
while a captured suffix retains its readers. The existing proxy stop publication
is the teardown authority. This host fixture substitutes only service work and
resource frees; the candidate and original predicates come from native source.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

root = Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT'])
backend = os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']
source = (root / 'tree' / backend / 'src/proxy.cc').read_text()
original = (Path(os.environ['LLAMINAR_NATIVE_REPLAY_ORIGINAL']) / 'src/proxy.cc').read_text()
output = root / 'replay'
output.mkdir(exist_ok=True)


def predicate(text):
    """Extract the complete service-loop expression from the pinned definition."""
    service = text[text.index('void* ncclProxyService('):]
    match = re.search(r'while\s*\((stop == PROXY_RUNNING.*?)\)\s*\{', service, re.S)
    if not match:
        raise RuntimeError('Missing native proxy service loop')
    return match.group(1)


store = re.search(r'__atomic_store_n\(&comm->proxyState->stop, 1, __ATOMIC_RELEASE\);',
                  source[source.index('ncclResult_t ncclProxyStop('):]).group(0)
program = '''#include <cassert>
#include <iostream>
struct Proxy { int stop; }; struct Comm { Proxy* proxyState; };
constexpr int PROXY_RUNNING=0, PROXY_STOP=1, PROXY_ABORT=2;
bool candidate(int stop,int npeers,Proxy*proxyState){return (''' + predicate(source) + ''');}
bool original(int stop,int npeers,Proxy*proxyState){return (''' + predicate(original) + ''');}
void retire(Comm*comm){''' + store + '''}
int main(){
for(int repeat=0;repeat<20;++repeat){
 Proxy state{0}; Comm comm{&state};
 // Closed peers after GROUP host abort: original enters resource cleanup early.
 assert(candidate(PROXY_ABORT,0,&state));
 assert(!original(PROXY_ABORT,0,&state));
 for(int poll=0;poll<100;++poll)assert(candidate(PROXY_ABORT,0,&state));
 // Existing last-owner stop must allow retirement without another peer message.
 retire(&comm); assert(!candidate(PROXY_ABORT,0,&state));
 // Normal stop and outstanding-peer draining retain their original semantics.
 for(int stop:{PROXY_RUNNING,PROXY_STOP,PROXY_ABORT})
  for(int peers:{0,1,4})
   assert(candidate(stop,peers,&state)==original(stop,peers,&state));
 state.stop=0;assert(!candidate(PROXY_STOP,0,&state));
 assert(candidate(PROXY_STOP,1,&state));
}
std::cout<<"PASS actual native proxy retirement:20 abort holds,20 original early frees,20 owner stops,normal stop and peer drain unchanged\\n";
}
'''
path = output / (backend + '-retirement.cpp')
path.write_text(program)
command = [os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'], '-std=c++17', '-O2', str(path), '-o', str(path.with_suffix(''))]
subprocess.run(command, check=True)
run = subprocess.run([str(path.with_suffix(''))], capture_output=True, text=True, check=True)
rows = [dict(backend=backend, returncode=run.returncode, output=run.stdout,
             source_sha256=hashlib.sha256(source.encode()).hexdigest(),
             original_sha256=hashlib.sha256(original.encode()).hexdigest())]
(output / 'retirement-result.json').write_text(json.dumps(rows, indent=2) + '\n')
print(run.stdout, end='')
