# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

import torch
import torch.nn as nn
from transformers import TrainerCallback
from superloss import SuperLoss

logger = logging.getLogger(__name__)

class ProportionalMixCallback(TrainerCallback):
    def __init__(self, train_dataset, cl_method, sample_difficlty):
        self.train_dataset = train_dataset
        self.cl_method = cl_method
        self.sample_difficlty = sample_difficlty
        self._last_stage = -1   # For curriculum in step-based training

    def on_epoch_begin(self, args, state, control, **kwargs):
        curr_epoch = state.epoch if state.epoch is not None else 0
        num_train_epochs = args.num_train_epochs
        self.train_dataset.curriculum(curr_epoch, num_train_epochs, self.cl_method, self.sample_difficlty)
        if int(os.environ.get('RANK')) == 0:
            print(f"Training dataset updated at epoch {state.epoch}")

    def on_step_begin(self, args, state, control, **kwargs):
        # Only works when args.max_steps is set (default args.max_steps is -1)
        if args.max_steps <= 0:
            return

        curr_step = state.global_step
        total_steps = args.max_steps

        # Only call curriculum when stage is updated
        chunk_size = total_steps / self.n_chunks
        stage = int(curr_step // chunk_size)
        if stage == self._last_stage:
            return  # In the same stage, no need to update the dataset
        self._last_stage = stage

        self.train_dataset.curriculum(curr_step, total_steps, self.cl_method, self.sample_difficulty)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, cl_cfg=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        self.cl_cfg = cl_cfg
        super().__init__(*args, **kwargs)
        self.setup_cl()

    def setup_cl(self):
        self.compute_loss = self.compute_loss_normal

        if self.cl_cfg.method == 'superloss':
            # Initialize SuperLoss calculator for Curriculum Learning
            self.super_loss = SuperLoss('sl', lam=self.cl_cfg.lam, C=self.cl_cfg.C, mode='avg')
            self.compute_loss = self.compute_loss_superloss
            logger.warning('************ Using SuperLoss! ************')

        elif self.cl_cfg.method in ['easy_to_hard', 'hard_to_easy']:
            self.sample_difficulty = torch.load(
                f'saves/sample_difficulty/{self.cl_cfg.data_name}/{self.cl_cfg.split}/{self.cl_cfg.difficulty_metric}.pt')
            self.add_callback(ProportionalMixCallback(self.train_dataset, self.cl_cfg.method, self.sample_difficulty))
            logger.warning('************ Using Easy-to-Hard or Hard-to-Easy Ordering! ************')

        else:
            logger.warning('************ No CL used! ************')

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     if self.cl_cfg is None or self.cl_cfg == 'none' or self.cl_cfg.method is None or self.cl_cfg.method == 'none':
    #         return self.compute_loss_normal(model, inputs, return_outputs)
    #     elif self.cl_cfg.method == 'superloss':
    #         return self.compute_loss_superloss(model, inputs, return_outputs)
    #     elif self.cl_cfg.method in ['easy_to_hard', 'hard_to_easy']:
    #         return self.compute_loss_normal(model, inputs, return_outputs)
    #     else:
    #         raise ValueError(f"{self.cl_cfg.method} should be None or in ['none', 'superloss', 'easy_to_hard', 'hard_to_easy']")

    def calculate_superloss(self, per_sample_loss):
        conf, tau, tau_adjusted = self.super_loss(per_sample_loss, None, None)
        tau = [tau] * per_sample_loss.shape[0]
        tau_adjusted = [tau_adjusted] * per_sample_loss.shape[0]
        sl_loss = per_sample_loss * conf

        return sl_loss


    # def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
    #     if self.cl_cfg is None or self.cl_cfg == 'none' or self.cl_cfg.method in ['none', 'superloss']:
    #         return super()._get_train_sampler(train_dataset)

    #     # Curriculum configuration
    #     num_chunks = self.args.num_train_epochs // 2  # 5 chunks for 10 epochs
    #     descending = True if self.cl_cfg.method == "hard_to_easy" else False

    #     # Current training epoch
    #     epoch = self.state.epoch if self.state.epoch is not None else 0
    #     max_stage = min(num_chunks, (epoch // 2) + 1)

    #     # Sort by difficulty and create curriculum chunks
    #     sorted_indices = torch.argsort(self.sample_difficulty, descending=descending)
    #     chunks = torch.chunk(sorted_indices, num_chunks)
    #     selected_indices = torch.cat(chunks[:max_stage]).tolist()

    #     # Create per-sample weights: 1.0 for included samples, 0.0 for excluded ones
    #     weights = torch.zeros(len(self.full_dataset), dtype=torch.float)
    #     for idx in selected_indices:
    #         weights[idx] = 1.0

    #     # Now create the sampler based on weights
    #     sampler = WeightedRandomSampler(weights=weights, num_samples=len(selected_indices), replacement=True)

    #     logger.info(f"[Curriculum] Epoch {epoch} — using {max_stage}/{num_chunks} chunks, "
    #                 f"total samples selected: {len(selected_indices)}")

    #     return sampler


    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
