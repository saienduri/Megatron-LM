#!/bin/bash
#

for file in "$1"/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        python3 parse_megatron_log.py $file
    fi
done

