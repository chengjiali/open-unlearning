import re
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


def wandb_setup(cfg):
    parts = cfg.get('task_name').strip().split('/')
    cl_method = parts[0]
    data, model, split, unlearn_method = parts[-1].split('_')

    project = 'Curriculum Unlearning'
    group = [data, split, model, unlearn_method]
    name = [data, split, model, unlearn_method]
    tags = [f"Data={data}", f"Split={split}", f"Model={model}", f"Unlearn_Method={unlearn_method}", f"CL={cl_method}"]

    if cl_method == 'none':
        name += [cl_method,]
    elif cl_method == 'superloss':
        C = re.search(r'C_(\d+)', parts[1]).group(1)
        lam = re.search(r'lam_([\d.]+)', parts[1]).group(1)
        name += [cl_method, C, lam]
        tags += [
            f"C={C}",
            f"Lam={lam}",
        ]
    elif cl_method in ['easy_to_hard', 'hard_to_easy']:
        difficulty_metric = parts[1]
        name += [cl_method, difficulty_metric]
        tags += [f"Metric={difficulty_metric}"]

    else:
        raise ValueError(f"Unknown CL method: {cl_method}")

    group = '-'.join(group)
    name = '-'.join(name)
    run_id = name
    wandb.init(project=project, group=group, name=name, 
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), 
               id=run_id, tags=tags, resume='allow')

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    wandb_setup(cfg)
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
