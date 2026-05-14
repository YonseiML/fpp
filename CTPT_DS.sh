data_root="${DATA_ROOT:-/path/to/datasets}"
DEVICE=0
exp_type=CTPT_DS
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_ctpt
lambda_term=20.0

exp_name="exp/${exp_type}"
SAVE_DIR=${exp_name}
csv_loc=${exp_name}/ctpt_ds.csv

mkdir -p ${SAVE_DIR}
cp -r clip ${SAVE_DIR}/
cp fpp_classification.py ${SAVE_DIR}/
cp ${exp_type}.sh ${SAVE_DIR}/


for seed in 0 1 2; do
    for testsets in A V R K; do
        echo "Running on test set: $testsets"
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=${SAVE_DIR}:$PYTHONPATH python ${SAVE_DIR}/fpp_classification.py ${data_root} \
            --test_sets ${testsets} \
            --csv_log ${csv_loc} \
            -a ${arch} \
            -b ${bs} \
            --gpu 0 \
            --tpt \
            --seed ${seed} \
            --ctx_init ${ctx_init} \
            --run_type ${run_type} \
            --lambda_term ${lambda_term}
    done
done
