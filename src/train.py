import sys
sys.path.append('/home/public/jcheng2/open-unlearning')
import re
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


def wandb_setup(cfg):
    parts = cfg.get('task_name').strip().split('/')
    cl_method = parts[0]
    if 'phi-1_5' in parts[-1]:
        parts[-1] = parts[-1].replace('phi-1_5', 'phi-1.5')
    data, model, split, unlearn_method = parts[-1].split('_')

    project = 'Curriculum Unlearning'
    group = [data, split, model, unlearn_method]
    name = [data, split, model, unlearn_method]
    tags = [f"Data={data}", f"Split={split}", f"Model={model}", f"Unlearn_Method={unlearn_method}", f"CL={cl_method}"]

    if cl_method == 'none':
        name += [cl_method,]
    elif cl_method in ['per_token_superloss', 'per_sample_superloss']:
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

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    wandb_setup(cfg)
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    # if eval_cfgs:
    #     evaluators = get_evaluators(
    #         eval_cfgs=eval_cfgs,
    #         template_args=template_args,
    #         model=model,
    #         tokenizer=tokenizer,
    #     )

    cl_cfg = trainer_cfg.cl
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
        cl_cfg=cl_cfg
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
