#!/bin/bash
cd build_v2_coverage/CMakeFiles/llaminar2_core.dir/loaders
gcov -l -p ModelLoader.cpp.gcno 2>&1 | grep -A 2 "^File.*ModelLoader.cpp$"
