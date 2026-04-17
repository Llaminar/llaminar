# EmbedJinjaTemplate.cmake
#
# Generator script for embedding a .jinja file into a C++ header as a raw
# string literal. Invoked via `cmake -P` from add_custom_command.
#
# Inputs (passed with -D):
#   JINJA_INPUT       Absolute path to the source .jinja file
#   HEADER_OUTPUT     Absolute path to the generated .h file
#   NAMESPACE         C++ namespace for the constant (e.g., llaminar2::qwen35)
#   SYMBOL            C++ identifier for the constant (e.g., kCommunityChatTemplate)
#   SOURCE_URL        Upstream source URL for attribution
#   LICENSE           License string (e.g., "MIT")
#
# The generated header exposes:
#   namespace <NAMESPACE> {
#       inline constexpr std::string_view <SYMBOL> = R"JINJA(... file contents ...)JINJA";
#   }
#
# The .jinja file must not contain the sequence `)JINJA"` (extremely unlikely
# for Jinja content). The check below fails the build if it does.

if(NOT DEFINED JINJA_INPUT OR NOT DEFINED HEADER_OUTPUT OR
   NOT DEFINED NAMESPACE OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "EmbedJinjaTemplate.cmake requires JINJA_INPUT, HEADER_OUTPUT, NAMESPACE, SYMBOL")
endif()

if(NOT DEFINED SOURCE_URL)
    set(SOURCE_URL "(not provided)")
endif()
if(NOT DEFINED LICENSE)
    set(LICENSE "(not provided)")
endif()

file(READ "${JINJA_INPUT}" JINJA_CONTENT)

# Guard against the closing delimiter appearing in the payload.
string(FIND "${JINJA_CONTENT}" ")JINJA\"" _clash)
if(NOT _clash EQUAL -1)
    message(FATAL_ERROR "Jinja file ${JINJA_INPUT} contains forbidden sequence ')JINJA\"' — pick a different raw-literal delimiter in EmbedJinjaTemplate.cmake")
endif()

get_filename_component(_input_name "${JINJA_INPUT}" NAME)

set(_header_content
"// ============================================================================
//  AUTO-GENERATED FILE — DO NOT EDIT BY HAND
//
//  Generated from: ${_input_name}
//  Source URL:     ${SOURCE_URL}
//  License:        ${LICENSE}
//
//  Regenerated at build time from the corresponding .jinja file by
//  cmake/EmbedJinjaTemplate.cmake. Edit the .jinja file instead of this one.
// ============================================================================

#pragma once

#include <string_view>

namespace ${NAMESPACE}
{
    inline constexpr std::string_view ${SYMBOL} = R\"JINJA(${JINJA_CONTENT})JINJA\";
}
")

file(WRITE "${HEADER_OUTPUT}" "${_header_content}")
