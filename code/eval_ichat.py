
import os
import sys
import json
import logging
from eval_ichat_utils import (
    parse_arguments,
    prepare_eval,
)
from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
)
from ichat_utils import IChatAPI

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()
    api = IChatAPI()

    for eval_name in args.eval_names:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)
        for subject in subjects:
            existing_results = None
            if os.path.exists(f'{args.save_path}/{subject}.jsonl'):
                logger.info(f"Results already exist: {args.save_path}/{subject}.jsonl")
                continue

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(api, few_shot_samples)

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                max_num_samples=args.max_num_samples,
                existing_results=existing_results,
            )

            metrics = None
            if args.setting not in ['cyclic'] and len(results) > 0:
                metrics = {'type': 'metric', 'data': {}}
                metrics['data']['accuracy'] = get_accuracy(results)
                metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                logger.info("Final report:")
                for key, value in metrics['data'].items():
                    logger.info(f"{key}: {value}")
            logger.info(_orange(f"Run completed: {subject}"))

            save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
            logger.info(f"Results saved: {subject}")


if __name__ == "__main__":
    main()

