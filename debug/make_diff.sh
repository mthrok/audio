#!/usr/bin/env bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${this_dir}"

diff -ru sox-14.4.2 debug > ../third_party/sox/patch/sox.patch

exit 0
