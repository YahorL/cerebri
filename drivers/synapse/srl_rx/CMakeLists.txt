# Copyright (c) 2025, CogniPilot Foundation
# SPDX-License-Identifier: Apache-2.0

zephyr_library_named(cerebri_synapse_srl_rx)

# we need to be able to include generated header files
zephyr_include_directories()

zephyr_library_sources(
  src/main.c
  )

add_dependencies(cerebri_synapse_srl_rx synapse_pb)
