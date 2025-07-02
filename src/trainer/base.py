# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

import torch
from transformers import TrainerCallback
from superloss import SuperLoss

logger = logging.getLogger(__name__)

class ProportionalMixCallback(TrainerCallback):
    def __init__(self, train_dataset, cl_method, sample_difficlty):
        self.train_dataset = train_dataset
        self.cl_method = cl_method
        self.sample_difficlty = sample_difficlty

    def on_epoch_begin(self, args, state, control, **kwargs):
        curr_epoch = state.epoch if state.epoch is not None else 0
        num_train_epochs = args.num_train_epochs
        self.train_dataset.curriculum(curr_epoch, num_train_epochs, self.cl_method, self.sample_difficlty)
        if int(os.environ.get('RANK')) == 0:
            print(f"Training dataset updated at epoch {state.epoch}")


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

        elif self.cl_cfg.method in ['easy_to_hard', 'hard_to_easy']:
            self.sample_difficulty = torch.load(
                f'saves/sample_difficulty/{self.cl_cfg.data_name}/{self.cl_cfg.split}/{self.cl_cfg.difficulty_metric}.pt')
            self.add_callback(ProportionalMixCallback(self.train_dataset, self.cl_cfg.method, self.sample_difficulty))

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

    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None or not has_length(self.train_dataset):
    #         return None

    #     # Build the sampler.
    #     if self.args.group_by_length:
    #         if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
    #             lengths = (
    #                 self.train_dataset[self.args.length_column_name]
    #                 if self.args.length_column_name in self.train_dataset.column_names
    #                 else None
    #             )
    #         else:
    #             lengths = None
    #         model_input_name = (
    #             self.processing_class.model_input_names[0] if self.processing_class is not None else None
    #         )
    #         return LengthGroupedSampler(
    #             self.args.train_batch_size * self.args.gradient_accumulation_steps,
    #             dataset=self.train_dataset,
    #             lengths=lengths,
    #             model_input_name=model_input_name,
    #         )

    #     else:
    #         return RandomSampler(self.train_dataset)
    
    # def get_batch_samples(self, epoch_iterator, num_batches):
    #     '''This function is called at every step'''
    #     if self.cl_cfg is None or self.cl_cfg == 'none' or self.cl_cfg.method in ['none', 'superloss']:
    #         return super().get_batch_samples(epoch_iterator, num_batches)

    #     new_train_dataloader = self.get_train_dataloader()
    #     epoch_iterator = iter(new_train_dataloader)
    #     return super().get_batch_samples(epoch_iterator, num_batches)


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


    # def get_train_dataloader(self):
    #     """
    #     Dataloader that supports curriculum learning.
    #     """

    #     if self.cl_cfg is None or self.cl_cfg == 'none' or self.cl_cfg.method == 'none':
    #         return super().get_trainer_dataloader()
        
    #     if self.full_dataset is None:
    #         print(os.environ.get('RANK'), 'Save a copy of the full dataset')
    #         self.full_dataset = self.train_dataset

    #     current_data_chunk = self.full_dataset

    #     # Apply curriculum here
    #     if self.cl_cfg.method not in ['superloss',]:
    #         num_chunks = self.args.num_train_epochs // 2 # 5 chunks for 10 epochs

    #         # Sort and chunk
    #         if self.cl_cfg.method == "hard_to_easy":
    #             descending = True
    #         elif self.cl_cfg.method == "easy_to_hard":
    #             descending = False
    #         else:
    #             raise NotImplementedError
    #         sorted_indices = torch.argsort(self.sample_difficulty, descending=descending)
    #         chunks = torch.chunk(sorted_indices, num_chunks)
            
    #         # Determine how many chunks to include at this epoch
    #         epoch = self.state.epoch if self.state.epoch is not None else 0
    #         max_stage = min(num_chunks, (epoch // 2) + 1)
    #         selected_indices = torch.cat(chunks[:max_stage]).tolist()

    #         msg = f"aaaaaa {os.environ.get('RANK')}, {len(self.full_dataset)}, {len(current_data_chunk)}, {epoch}, {selected_indices}"
    #         logger.info(msg)

    #         if is_datasets_available() and isinstance(current_data_chunk, datasets.Dataset):
    #             current_data_chunk = current_data_chunk.select(selected_indices)
    #         elif isinstance(current_data_chunk, torch.utils.data.Dataset):
    #             current_data_chunk = Subset(current_data_chunk, selected_indices)
    #         else:
    #             raise NotImplementedError
        
    #     # Overwrite the train_dataset, since self._get_train_sampler() uses self.train_dataset to get dataset length
    #     self.train_dataset = current_data_chunk

    #     # Original transormers.trainer code
    #     data_collator = self.data_collator
    #     if is_datasets_available() and isinstance(current_data_chunk, datasets.Dataset):
    #         current_data_chunk = self._remove_unused_columns(current_data_chunk, description="training")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    #     dataloader_params = {
    #         "batch_size": self._train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     if not isinstance(current_data_chunk, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = self._get_train_sampler()
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker
    #         dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

    #     return self.accelerator.prepare(DataLoader(current_data_chunk, **dataloader_params))

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
