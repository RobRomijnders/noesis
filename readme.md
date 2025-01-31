# NoEsis: A Modular LLM with Differentially Private Knowledge Transfer

This is the Github repo that goes with the NoEsis project. The paper can be found at [arXiv](https://arxiv.org/).

# Structure
Important abstractions made in the code and important changes with respect to default DP fine tuning:

  * `dp_trainer.py` subclasses the normal trainer.py from Huggingface and adds the DP functionality. There's an object PrivacyArgs that is passed to the Trainer object that contains the DP parameters. The constructor will grab dataloader, optimizer, criterion and model from the Trainer object and will create a PrivacyEngine object from Opacus. The wrapped objects will take care of privacy accounting and the usual clipping and noising of stochastic gradients.
  * `moe.py` subclasses the model construction of the CodeT5+ model. Most notably, we add support for the Mixture-of-LoRA architecture. When corresponding flags are set, one can also train with learnable prompts or prefixes. Important hyperparameters are num_prompt_tokens and num_prefix_tokens for the number of learnable prompts or prefixes, respectively; also there is rank_domain and rank_common, which are the LoRA ranks of the domain-specific and common parameters, respectively.
  * `util_model.py` contains the model loading and saving functions. The load_model function will load the model, but with some exceptions, as the upstream CodeT5+ model does not have experts, so they are escaped in the load_dict() call and initialized from scratch.
  * `util_data.py` contains the data loading and processing functions. The load_data function will load the dataset and tokenize it. The get_dataloaders function will create the dataloaders for the training and validation sets. Note that to ensure DP, each epoch can use each document only once (leading to the definition of document-level DP). Therefore, we wrote a custom data collator that samples a different block of code per document on each new epoch.
  * `util.py` has a few string manipulation functions. For the multi-domain training, one can define a comma-separated list of programming languages. These string manipulation functions will take care of the splitting and joining of the languages.

# Running the code

A typical call for training the model is as follows:

```bash
python3 main_tune.py \
    --batch-size-per-replica=16 \
    --cache-data=cache_data/ \
    --datasets='java,python,go' \
    --epochs=12 \
    --freeze-backbone \
    --load=Salesforce/codet5p-220m \
    --num-experts=3 \
    --num-prompt-tokens=32 \
    --rank-domain=-1 \
    --save-dir=saved_models/my_experiment \
    --seed=123 \
    --target-epsilon=1.
  ```

  We include the `scripts/run_dpfirst_pt.sh` shell script to execute this command and relevant flags in one go.
  Using `scripts/run_ftsecond_init__frompt.sh`, one can train the domain-specific parameters in the second tage

# Information

More information, please contact authors