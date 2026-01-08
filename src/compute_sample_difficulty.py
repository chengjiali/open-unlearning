import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals.compute_sample_difficulty import EvaluatorComputeSampleDifficulty


# import debugpy
# debugpy.listen(('localhost', 5678))
# print('waiting for debugger attach...')
# debugpy.wait_for_client()
# debugpy.breakpoint()


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    task_name = cfg.get('task_name')

    if 'tofu' in task_name:
        dataset_name = 'tofu'
        data_split = cfg.get('forget_split')

    elif 'muse' in task_name:
        dataset_name = 'muse'
        data_split = cfg.get('data_split')

    elif 'wmdp' in task_name:
        dataset_name = 'wmdp'
        data_split = cfg.get('data_split')

    # eval_cfgs = cfg.eval
    # for evaluator_name, eval_cfg in eval_cfgs.items():
    import os
    os.makedirs(f"saves/sample_difficulty/{dataset_name}", exist_ok=True)
    eval_args = {
        "template_args": template_args,
        "model": model,
        "tokenizer": tokenizer,
        "output_dir": f"saves/sample_difficulty/{dataset_name}/{task_name}"
    }
    evaluator = EvaluatorComputeSampleDifficulty(dataset_name, data_split, cfg, **eval_args)
    _ = evaluator.compute_sample_difficulty(**eval_args)


if __name__ == "__main__":
    main()
