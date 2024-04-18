import os
import torch
from transformers import Trainer
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class Trainer(Trainer):

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

	### [commented] ###
        # if "model_path" in kwargs:
        #     resume_from_checkpoint = kwargs.pop("model_path")
        #     warnings.warn(
        #         "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
        #         "instead.",
        #         FutureWarning,
        #     )
        # if len(kwargs) > 0:
        #     raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

	### [commented] ###
        # Model re-init 
        # model_reloaded = False
        # if self.model_init is not None:
        #     # Seed must be set before instantiating the model when using model_init.
        #     enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        #     self.model = self.call_model_init(trial)
        #     model_reloaded = True
        #     # Reinitializes optimizer and scheduler
        #     self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

	### [commented] ###
        # If model was re-initialized, put it on the right device and update self.model_wrapped
        # if model_reloaded:
        #     if self.place_model_on_device:
        #         self._move_model_to_device(self.model, args.device)
        #     self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
	### [commented] ###
        # if args.push_to_hub:
        #     try:
        #         # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
        #         hf_hub_utils.disable_progress_bars()
        #         return inner_training_loop(
        #             args=args,
        #             resume_from_checkpoint=resume_from_checkpoint,
        #             trial=trial,
        #             ignore_keys_for_eval=ignore_keys_for_eval,
        #         )
        #     finally:
        #         hf_hub_utils.enable_progress_bars()
        # else:

	### [TODO] See if we need to do the multi-stage training here as the flows might be different.
	### And we would like to update the new dataloader
	return inner_training_loop(
	    args=args,
	    resume_from_checkpoint=resume_from_checkpoint,
	    trial=trial,
	    ignore_keys_for_eval=ignore_keys_for_eval,
	)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


        if self.state.global_step % 10 == 0:
            logger.info(f"loss: {outputs['loss'].item()} | acc: {outputs['acc']}")
            self.log({"loss": outputs['loss'].item(), "acc": outputs['acc'].item()})
            if outputs.get('logs', None):
                for k, v in outputs['logs'].items():
                    logger.info(f"{k}: {v.item()}")
                    self.log({f"{k}": v.item()})

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, **kwargs):
        """ Discard the original argument of `state_dict`, since it's from entire wrapped model.
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}. The model checkpoint is an encoder for huggingface, not a wrapping model.")

        model = self.model.get_encoder()
        self.model.encoder.save_pretrained(
            output_dir, state_dict=model.state_dict(), safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))