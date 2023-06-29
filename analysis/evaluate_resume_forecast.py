"""Compute the test perplexity of each model on the resume data."""
import argparse
import os
import pdb
import torch
import torch.nn.functional as F
from fairseq import utils

from fairseq.models.transformer import TransformerModel
# from fairseq.models.bag_of_jobs import BagOfJobsModel
# from fairseq.models.lstm import LSTMModel
# from fairseq.models.regression import RegressionModel
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--binary-data-dir", 
                    type=str,
                    help="Location of binarized data.")
parser.add_argument("--save-dir", 
                    type=str,
                    help="Location of saved model checkpoint.")
parser.add_argument("--model-name", 
                    type=str,
                    default='career',
                    help="Model name (must be one of: 'career', 'bag-of-jobs',"
                         " 'regression', or 'lstm').")
parser.add_argument("--num-samples", 
                    type=int,
                    default=1000,
                    help="Number of Monte Carlo samples.")
parser.add_argument('--no-covariates', 
                    action='store_true',
                    help="Load model that does not use covariates.")
args = parser.parse_args()

model_path = os.path.join(
  args.save_dir, 
  'forecast-resumes/{}{}'.format(
    args.model_name, '_no_covariates' if args.no_covariates else ''))
binary_data_path = os.path.join(args.binary_data_dir, "resumes")

# Load model.
if args.model_name == 'career':
  model = TransformerModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)

# Move model to GPU and set to eval mode.
if torch.cuda.is_available():
  model.cuda()
model.eval()
model.model = model.models[0]
two_stage = model.model.decoder.args.two_stage

# Load test data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=100,).next_epoch_itr(shuffle=False)

cutoff_year = 2020  # last year to be included in the training data.
cutoff_index = model.task._year_dictionary.index(str(cutoff_year))

if args.model_name in ['career', 'regression']:
  year_weights = model.model.decoder.embed_year.weight.detach().cpu().numpy()
  last_year_weights = year_weights[cutoff_index, :]
  year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_year.weight.data = torch.from_numpy(
    year_weights).cuda()
else:
  current_year_weights = (
    model.model.decoder.embed_current_year.weight.detach().cpu().numpy())
  last_year_weights = current_year_weights[cutoff_index, :]
  current_year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_current_year.weight.data = (
    torch.from_numpy(current_year_weights).cuda())
  context_year_weights = (
    model.model.decoder.embed_context_year.weight.detach().cpu().numpy())
  last_year_weights = context_year_weights[cutoff_index, :]
  context_year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_context_year.weight.data = (
    torch.from_numpy(context_year_weights).cuda())

num_samples = args.num_samples
max_possible_year = max(
  [int(x) for x in model.task._year_dictionary.symbols if x.isdigit()])

all_nll = {}

for eval_index, sample in enumerate(itr):
  print("Evaluating {}/{}...".format(eval_index, itr.total))
  with torch.no_grad():
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    
    # Compute log probs for the sample.
    sample['net_input']['prev_output_tokens'] = sample[
      'net_input']['src_tokens']
    del sample['net_input']['src_tokens']

    # Need to simulate all outcomes beyond 2021.
    batch_size = sample['target'].size(0)
    # Sample trajectories for each individual in the batch.
    for batch_ind in range(batch_size):
      all_years = np.array(
        [model.task._year_dictionary.string([x]) 
         for x in sample['net_input']['years'][batch_ind]])
      all_years = np.array(
        [int(x) for x in all_years if len(x) > 0 and x != '<pad>'])
      # Only forecast for individuals who start before the cutoff and have
      # observations through the maximum possible year.
      if max(all_years) == max_possible_year and min(all_years) <= cutoff_year:
        first_simulated_index = np.where(all_years > cutoff_year)[0][0]
        # Copy all pre-cutoff information for each sample to predict the first 
        # post-cutoff job.
        prev_tokens = sample['net_input']['prev_output_tokens'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        years = sample['net_input']['years'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        educations = sample['net_input']['educations'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        locations = sample['net_input']['locations'][batch_ind].repeat(
          [num_samples, 1])
        # Simulate each year, one-by-one.
        num_unseen_years = sum(all_years >= cutoff_year)
        for simulated_year in range(first_simulated_index, 
                                    first_simulated_index + num_unseen_years):
          true_job = sample['target'][batch_ind][simulated_year].item()
          output = model.model.decoder.forward(
            prev_tokens, years=years, educations=educations, locations=locations)
          lprobs = model.model.get_normalized_probs(
            output, log_probs=True, two_stage=two_stage, 
            prev_tokens=prev_tokens)[:, -1]
          # Sample next job from distribution.
          next_job_samples = torch.multinomial(lprobs, num_samples=1)
          average_prob = lprobs.mean(0)
          all_nll.setdefault(simulated_year, []).append(-np.log(average_prob[true_job].item()))
          # Update for next year.
          prev_tokens = torch.cat([prev_tokens, next_job_samples], -1)
          years = sample['net_input']['years'][batch_ind][
            :(simulated_year + 2)][None].repeat([num_samples, 1])
          educations = sample['net_input']['educations'][batch_ind][
            :(simulated_year + 2)][None].repeat([num_samples, 1])

perplexities = {year: np.exp(np.mean(nll)) for year, nll in all_nll}

print("..................")
print("Test-set results for {}{}, model loaded from '{}'".format(
  args.model_name, " (no_covariates)" if args.no_covariates else "",
  model_path))
print("..................")
for year, ppl in perplexities.items():
  print("Year {} perplexity: {:.2f}".format(year, ppl))
print("..................")
