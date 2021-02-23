#/usr/bin/env bash

debug_func() {
    prefix="$1"
    script="$2"
    
    cd test
    wrk_dir=./debug_fileobj/"${prefix}"
    rm -rf "${wrk_dir}"
    mkdir -p "${wrk_dir}"
    counter=0
    while true; do
        counter=$((counter + 1))
        echo "${counter}-th" > "${wrk_dir}/log"
        export TORCHAUDIO_TEST_TEMP_DIR="${wrk_dir}/data"
        pytest "torchaudio_unittest/backend/sox_io/${script}" >> "${wrk_dir}/log" 2>&1
        if [ $? -ne 0 ] ; then
            break;
        fi
    done
}

debug_save() {
    debug_func save save_test.py
}

debug_load() {
    debug_func load load_test.py
}

debug_info() {
    debug_func info info_test.py
}


# (debug_save &)
(debug_load &)
# (debug_info &)

# watch tail test/debug_fileobj/*/failure/log
