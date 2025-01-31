"""Running the MIA attack on the mixed-domain models."""
# pylint: disable=line-too-long,invalid-name
import argparse
from datetime import datetime
import os
import random

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from NoEsis import constants, util_data, util_model, util_mia

EXPERTS = ['go', 'java', 'python']


def load_dataset_mia(args, datatype):
  """Loads the datasets for the MIA attack."""
  assert datatype in ['member', 'nonmember']

  cachedir = f"/tmp/dataset_cache_v01_{args.datasets}_{args.max_target_len}"
  os.makedirs(cachedir, exist_ok=True)

  # Set all random seeds
  np.random.seed(args.seed)
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  # Member data
  if datatype == 'member':
    args.use_dedup = False
    cachedir_train = os.path.join(cachedir, 'train')
    if os.path.exists(cachedir_train):
      dataset_obj = datasets.load_from_disk(cachedir_train)
      print(f'  ==> Loaded {len(dataset_obj)} samples from cache at {cachedir_train}')
    else:
      dataset_obj = util_data.provision_dataset(args, split='train', testtime=True)
      os.makedirs(cachedir_train, exist_ok=True)
      dataset_obj.save_to_disk(cachedir_train)
      print(f'  ==> Loaded {len(dataset_obj)} samples  ==> Saved to {cachedir_train}')

  elif datatype == 'nonmember':
    # Non-member data
    args.use_dedup = False
    cachedir_dedup = os.path.join(cachedir, 'dedup')
    if os.path.exists(cachedir_dedup):
      dataset_obj = datasets.load_from_disk(cachedir_dedup)
      print(f'  ==> Loaded {len(dataset_obj)} samples from cache at {cachedir_dedup}')
    else:
      dataset_obj = util_data.provision_dataset(args, split='test', testtime=True)
      dataset_obj.save_to_disk(cachedir_dedup)
      print(f'  ==> Loaded {len(dataset_obj)} samples  ==> Saved to {cachedir_dedup}')
  else:
    raise ValueError(f"Invalid datatype: {datatype}")

  return dataset_obj

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run MIA attack on mixed-domain models.')
  parser.add_argument('--datatype', type=str, choices=['member', 'nonmember'], help="Specify the data type: 'member' or 'nonmember'")
  parser.add_argument('--language', type=str, choices=['java', 'python', 'go'], help="Specify the language: 'java' or 'python'")
  parser.add_argument('--deviceid', type=int, default=0, help="Specify the device id for GPU usage")

  args = parser.parse_args()

  datatype = str(args.datatype).lower()
  language_str = str(args.language).lower()
  deviceid = int(args.deviceid)
  assert 0 <= deviceid <= 7

  name_checkpoint = "Salesforce/codet5p-220m"
  device = f"cuda:{deviceid}" # for GPU usage or "cpu" for CPU usage

  # load the model
  class Args:  # pylint: disable=too-few-public-methods
      """Model parameters."""
      max_target_len = 512
      load = name_checkpoint
      datasets = language_str
      seed = 123

  args = Args()

  ckpt_dir_nonprivate = '/root/code/NoEsis/NoEsis/saved_models/ftsecond_fromrc_nonprivjava,python,go_3_0_seed130/go-java-python_num3_seed130_6k01d4ct/final_checkpoint'  # 20250123
  model_nonprivate = util_model.load_noesis_model(ckpt_dir=ckpt_dir_nonprivate)

  ckpt_dir_noesis = '/root/code/NoEsis/NoEsis/saved_models/ftsecond_frompt_java,python,go_3_0_seed130/go-java-python_num3_seed130_ns4eh692/final_checkpoint'  # Stepwise, eps=1.0, from pt32
  model_private = util_model.load_noesis_model(ckpt_dir=ckpt_dir_noesis)

  if language_str == 'python':
    ckpt_dir_dedup = '/root/code/NoEsis/NoEsis/saved_models/20250109__dedup_python_32latents_120epochs'  # 120e, rc32
  elif language_str == 'java':
    ckpt_dir_dedup = '/root/code/NoEsis/NoEsis/saved_models/20250109__java_dedup__16latents__120epochs'  # 120 epochs, rc16
  elif language_str == 'go':
    ckpt_dir_dedup = '/root/code/NoEsis/NoEsis/saved_models/20250109__dedup_python_32latents_120epochs'  # this is a python model, but just to have a placeholder
  else:
    raise ValueError(f"Invalid {language_str}, only Python and Java for now")
  model_reference = util_model.load_noesis_model(ckpt_dir=ckpt_dir_dedup)

  # Load the data
  dataset_object = load_dataset_mia(args, datatype)

  # Set the random seeds
  util_mia.set_seed(args.seed)

  # Run the MIA
  batch_size = 32

  # Run membership inference attack, collect all logits from training set and test set to see the difference
  model_private = model_private.to(device).eval()
  model_nonprivate = model_nonprivate.to(device).eval()
  model_reference = model_reference.to(device).eval()

  num_bins = 1000
  bins = np.linspace(-6, 1, num_bins + 1)

  def make_histogram(dataset_obj, expert_str):  # pylint: disable=too-many-locals
    """Creates the histogram for the MIA attack.

    Each histogram matrix is a stack of five histograms:
    0: [noesis] not comparing against reference
    1: [noesis] comparing against reference
    2: softmax logits of the actual reference
    3: [nonprivate] not comparing against reference
    4: [nonprivate] comparing against reference
    """
    values_priv = []
    values_priv_ref = []
    values_ref = []
    values_nonpriv = []
    values_nonpriv_ref = []

    expert_ids = constants.LANGUAGE_MAP[expert_str]*torch.ones((batch_size), dtype=torch.long)
    expert_ids = expert_ids.long().to(device)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

    print(f"Running MIA {datatype} data, for language {language_str} via expert {expert_str} ({os.getpid()}/{os.getppid()})")
    for num_batch, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):  # pylint: disable=unused-variable
      decoder_input_ids = torch.stack(batch["decoder_input_ids"], axis=-1).to(device)
      labels = torch.stack(batch["labels"], axis=-1).to(device)
      decoder_attention_mask = torch.stack(batch["decoder_attention_mask"], axis=-1).to(device)

      local_batch_size = decoder_input_ids.size(0)
      if local_batch_size < batch_size:
        expert_ids = constants.LANGUAGE_MAP[expert_str]*torch.ones((local_batch_size), dtype=torch.long)
        expert_ids = expert_ids.long().to(device)

      with torch.no_grad():
        logits_private = model_private(
          expert_ids=expert_ids,
          decoder_input_ids=decoder_input_ids,
          decoder_attention_mask=decoder_attention_mask)

        logits_nonprivate = model_nonprivate(
          expert_ids=expert_ids,
          decoder_input_ids=decoder_input_ids,
          decoder_attention_mask=decoder_attention_mask)

        logits_reference = model_reference(
          expert_ids=None,  # ignored on solo model
          decoder_input_ids=decoder_input_ids,
          decoder_attention_mask=decoder_attention_mask)

        # NoEsis softmax
        log_softmax = torch.nn.functional.log_softmax(logits_private, dim=-1)
        softmax_g_private = torch.gather(input=log_softmax, dim=2, index=torch.abs(labels).unsqueeze(-1)).squeeze(-1).detach()

        # Nonprivate softmax
        log_softmax_nonprivate = torch.nn.functional.log_softmax(logits_nonprivate, dim=-1)
        softmax_g_nonprivate = torch.gather(input=log_softmax_nonprivate, dim=2, index=torch.abs(labels).unsqueeze(-1)).squeeze(-1).detach()

        # Dedup softmax
        log_softmax_reference = torch.nn.functional.log_softmax(logits_reference, dim=-1)
        softmax_g_reference = torch.gather(input=log_softmax_reference, dim=2, index=torch.abs(labels).unsqueeze(-1)).squeeze(-1).detach()

        # ################
        mask = (labels >= 0).detach().clone().float()
        mask_sum = mask.sum(axis=1)

        # Hist NoEsis
        softmax_mean = torch.sum(softmax_g_private * mask, axis=1) / mask_sum
        values_priv.append(softmax_mean.cpu().numpy())

        softmax_m_dedup = torch.sum((softmax_g_private - softmax_g_reference) * mask, axis=1) / mask_sum
        values_priv_ref.append(softmax_m_dedup.cpu().numpy())

        # Hist Reference
        softmax_reference = torch.sum(softmax_g_reference * mask, axis=1) / mask_sum
        values_ref.append(softmax_reference.cpu().numpy())

        # Hist Nonprivate
        softmax_mean_nonprivate = torch.sum(softmax_g_nonprivate * mask, axis=1) / mask_sum
        values_nonpriv.append(softmax_mean_nonprivate.cpu().numpy())

        softmax_m_dedup_nonprivate = torch.sum((softmax_g_nonprivate - softmax_g_reference) * mask, axis=1) / mask_sum
        values_nonpriv_ref.append(softmax_m_dedup_nonprivate.cpu().numpy())

    # Save the data
    current_time = datetime.now().strftime("%y%m%d")
    filename = f"/root/histogramdata/{current_time}__{language_str}__{expert_str}__{datatype}.npz"
    print(f"Saving histograms to {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.savez(
      filename,
      values_priv=np.concatenate(values_priv, axis=0),
      values_priv_ref=np.concatenate(values_priv_ref, axis=0),
      values_ref=np.concatenate(values_ref, axis=0),
      values_nonpriv=np.concatenate(values_nonpriv, axis=0),
      values_nonpriv_ref=np.concatenate(values_nonpriv_ref, axis=0))

  for expert_str in EXPERTS:
    if expert_str.lower().strip() == language_str.lower().strip():
      print(f"Skipping {expert_str} as it is the same as the language")
      continue
    make_histogram(dataset_object, expert_str=expert_str)
