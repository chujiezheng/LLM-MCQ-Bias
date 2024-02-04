
names=()
for task in mmlu arc; do
    for num_shot in 5 0; do
        for setting in perm noid shuffle_both; do
            names[${#names[*]}]="${task},${num_shot},${setting}"
        done
    done
done
for task in csqa; do
    for num_shot in 5 0; do
        for setting in cyclic noid shuffle_both; do
            names[${#names[*]}]="${task},${num_shot},${setting}"
        done
    done
done
for task in mmlu; do
    for num_shot in 5 0; do
        for setting in move_a move_b move_c move_d; do
            names[${#names[*]}]="${task},${num_shot},${setting}"
        done
    done
done

pretrained_model_path=${HF_MODELS}/huggyllama/llama-7b

python eval_clm.py \
    --pretrained_model_path ${pretrained_model_path} \
    --eval_names ${names[*]} 

