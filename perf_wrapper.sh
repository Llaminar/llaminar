#!/bin/bash
# Wrapper script to run llaminar2 under perf for each MPI rank
# Each rank gets its own perf.data file

RANK=${OMPI_COMM_WORLD_RANK:-0}
OUTPUT_FILE="perf_rank${RANK}.data"

# Use higher sampling frequency (-F 999) and cpu-clock for software event
exec perf record -F 999 -e cpu-clock -g -o "$OUTPUT_FILE" -- "$@"


##################
## HOW TO USE THIS
##################

###### How to run perf record:

# cd /workspaces/llaminar && \
# rm -f perf_rank*.data && \
# export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false && \
# export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 && \
# export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false && \
# export OMPI_MCA_mpi_leave_pinned=1 OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1 && \
# export LLAMINAR_LOG_LEVEL=ERROR && \
# sudo -E mpirun --allow-run-as-root -np 2 --bind-to socket --map-by socket \
#   ./perf_wrapper.sh ./build_v2_release/llaminar2 \
#   -m ./models/Qwen2.5-14B-Instruct.Q8_0.gguf \
#   --chat-single -p "Tell me about Australia." \
#   --activation-precision q8_1


###### To analyze the perf data later, do:

# cd /workspaces/llaminar && echo "=== BY SHARED OBJECT ===" && \
# sudo perf report -i perf_rank0.data --stdio --no-children --sort=dso -g none 2>/dev/null | grep -E "^\s+[0-9]+\.[0-9]+%" | head -15