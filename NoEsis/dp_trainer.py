"""Specializer trainer that can handle DP for MoE."""
import copy
from dataclasses import dataclass, field
from typing import Optional

import opacus
import torch
import transformers

from NoEsis import logger


@dataclass
class DpArguments:
    """Dataclass for all arguments related to Differential Privacy."""

    per_sample_max_grad_norm: Optional[float] = field(default=None, metadata={
        "help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={
        "help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=None, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=float(1E-6), metadata={
        "help": "Target delta, defaults to 1/N"
    })
    disable_dp: bool = field(default=False, metadata={
        "help": "Disable DP training."
    })
    secure_mode: bool = field(default=False, metadata={
        "help": "Use secure mode for DP-SGD."
    })
    max_physical_per_device_train_batch_size: Optional[int] = field(default=None, metadata={
        "help": "Maximum physical batch size per device for training."
    })
    poisson_sampling: bool = field(default=False, metadata={
        "help": "Use Poisson sampling for DP-SGD."
    })
    grad_sample_mode: str = field(default="hooks", metadata={
        "help": "Mode for per-sample gradients. one of ['hooks', 'ghost']"
    })


class NoEsisTrainer(transformers.Trainer):
    """Specializer trainer that can handle DP for MoE."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        privacy_args: DpArguments,
        **kwargs) -> None:
        self.train_args: transformers.TrainingArguments = kwargs.pop("args")

        model.shared.weight.requires_grad = False
        model.lm_head.weight.requires_grad = False
        do_param_share = model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr()
        print(f"Parameter sharing is active: {do_param_share}.")

        self.privacy_args = privacy_args
        self.privacy_engine = opacus.PrivacyEngine(secure_mode=self.privacy_args.secure_mode)

        acc_sum, n_acc_sum = 0, 0
        def compute_metrics(eval_pred, compute_result=False):
            """Compute metrics for the evaluation."""
            logits, labels = eval_pred
            accuracy = (torch.argmax(logits, dim=-1) == labels).float().mean().item()

            # Take non-local floats to accumulate the accuracy
            nonlocal acc_sum, n_acc_sum
            acc_sum += accuracy
            n_acc_sum += 1
            if compute_result:
                return {"accuracy": acc_sum / n_acc_sum}
            return {"accuracy": accuracy}

        super().__init__(
            args=self.train_args,
            model=model,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            **kwargs)

        super().create_optimizer()
        self.non_dp_optimizer = self.optimizer

        config_model = copy.deepcopy(model.config)
        self.model = model.train()

        config_model.save_pretrained(self.train_args.output_dir)

        self.dataloader_overwrite = super().get_train_dataloader()
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.privacy_args.disable_dp:
            print("INFO: DP is disabled.")
            return

        print(f"Type of model after calling Trainer() constructor: {type(model)}")
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(model)

        print(f"Privatizing optimizer with mode {privacy_args.grad_sample_mode}")
        if self.privacy_args.target_epsilon < 0:
            (
                self.model_gc, self.optimizer_gc, self.criterion_gc, self.dataloader_gc
                ) = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.non_dp_optimizer,
                    criterion=torch.nn.CrossEntropyLoss(),
                    poisson_sampling=False,
                    data_loader=self.dataloader_overwrite,
                    noise_multiplier=self.privacy_args.noise_multiplier,
                    max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                    grad_sample_mode=privacy_args.grad_sample_mode)
        else:
            (
                self.model_gc, self.optimizer_gc, self.criterion_gc, self.dataloader_gc
                ) = self.privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=self.non_dp_optimizer,
                    criterion=self.criterion,
                    poisson_sampling=False,
                    epochs=self.train_args.num_train_epochs,
                    data_loader=super().get_train_dataloader(),
                    target_epsilon=self.privacy_args.target_epsilon,
                    target_delta=self.privacy_args.target_delta,
                    max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                    grad_sample_mode=privacy_args.grad_sample_mode)

        # Re-attribute config as GradSampleModule wraps model, causing WandB not to find the config
        self.model_gc.config = config_model

        self.model = self.model_gc
        self.optimizer = self.optimizer_gc
        self.criterion = self.criterion_gc
        self.dataloader_overwrite = self.dataloader_gc

    def get_train_dataloader(self):
        # Call super().get_train_dataloader() to get the original dataloader
        # and wrap it with the privacy in __init__
        return self.dataloader_overwrite

    def compute_loss(self, model, inputs, return_outputs=False):
        # Overwrites transformers:trainer.py::Trainer.compute_loss
        logits = model(**inputs)

        loss = self.criterion(logits.transpose(1, 2), inputs["labels"])
        return (loss, {'logits': logits}) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Saves the model with super().save_model and logs the current epsilon."""
        super().save_model(output_dir, _internal_call)

        if not self.privacy_args.disable_dp:
            eps = self.privacy_engine.get_epsilon(delta=float(self.privacy_args.target_delta))
            logger.info(f"CURRENT EPSILON SPENT [log]: {eps:.2f}")
