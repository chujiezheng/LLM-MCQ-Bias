
models=(
    "llama-7b"
    "llama-13b"
    "llama-30b"
    "llama-65b"
    "Llama-2-7b-hf"
    "Llama-2-13b-hf"
    "Llama-2-70b-hf"
    "Llama-2-7b-chat-hf"
    "Llama-2-13b-chat-hf"
    "Llama-2-70b-chat-hf"
    "vicuna-7b-v1.3"
    "vicuna-13b-v1.3"
    "vicuna-33b-v1.3"
    "vicuna-7b-v1.5"
    "vicuna-13b-v1.5"
    "falcon-7b"
    "falcon-40b"
    "falcon-7b-instruct"
    "falcon-40b-instruct"
)

for num_shot in 0 5; do
    for task in "mmlu" "arc"; do
        paths=()
        for model in ${models[*]}; do
            paths[${#paths[*]}]="results_${task}/${num_shot}s_${model}/${task}_perm"
        done
        for debias_fn in "simple" "full"; do
            python debias_base.py --task ${task} --load_paths ${paths[*]} --debias_fn ${debias_fn}
        done
    done
done

for num_shot in 0 5; do
    for task in "csqa"; do
        paths=()
        for model in ${models[*]}; do
            paths[${#paths[*]}]="results_${task}/${num_shot}s_${model}/${task}_cyclic"
        done
        for debias_fn in "simple"; do
            python debias_base.py --task ${task} --load_paths ${paths[*]} --debias_fn ${debias_fn}
        done
    done
done

models=(
    "ichat"
)

for num_shot in 0 5; do
    for task in "mmlu" "arc" "csqa"; do
        paths=()
        for model in ${models[*]}; do
            paths[${#paths[*]}]="results_${task}/${num_shot}s_ichat/${task}_cyclic"
        done
        for debias_fn in "simple"; do
            python debias_base.py --task ${task} --load_paths ${paths[*]} --debias_fn ${debias_fn}
        done
    done
done
