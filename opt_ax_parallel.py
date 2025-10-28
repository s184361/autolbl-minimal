import argparse
import json
import os
import torch
from torch.nn import functional as F
import wandb
import pandas as pd
import numpy as np
import random
from typing import Dict, Any, NamedTuple, Union, Iterable, Set
from collections import defaultdict
from time import time
from random import randint

# Ax imports
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.core.base_trial import TrialStatus, BaseTrial
from ax.core.trial import Trial
from ax.core.runner import Runner
# Add these new imports for Metric and related classes
from ax.core.metric import Metric
from ax.core.data import Data
from ax.utils.common.result import Ok, Err
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
# Add these new imports for experiment setup
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.objective import Objective
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig

# Local imports
from run_any2 import run_any_args
from utils.check_labels import *
from utils.wandb_utils import compare_plot as compare_wandb_plot
from transformers import BertTokenizerFast
try:
    from ultralytics.utils.loss import E2EDetectLoss as DETRLoss
except ImportError:
    try:
        from ultralytics.models.utils.loss import DETRLoss
    except ImportError:
        print("Warning: Could not import DETRLoss. Using placeholder.")
        DETRLoss = None
global global_optimizer

def parse_args():  # Fixed typo in function name
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--randomize', action='store_true')  # Changed to flag
    parser.add_argument('--initial_prompt', type=str, default='defect')
    parser.add_argument('--ds_name', type=str, default='tires')
    parser.add_argument('--model', type=str, default='Florence')
    parser.add_argument('--optimizer', type=str, default='ax')
    parser.add_argument('--encoding_type', type=str, default='bert')
    return parser.parse_args()

class PromptOptimizer:
    def __init__(self,
                config_path='config.json',
                encoding_type='bert',
                randomize=False,
                model="Florence",
                optimizer="ax",
                ds_name="tires",
                initial_prompt="defect"):
        """
        Initialize the prompt optimizer.
        
        Args:
            config_path: Path to configuration JSON file
            encoding_type: Type of encoding to use ('bert' or 'ascii')
            randomize: Whether to randomize initial prompt tokens
            model: Model to use for inference
            optimizer: Optimization strategy to use
            ds_name: Dataset name
            initial_prompt: Starting text prompt
        """
        torch.cuda.empty_cache()
        wandb.login()
        
        # Store configuration parameters
        self.randomize = randomize  # Fixed: now uses parameter
        self.initial_prompt = initial_prompt
        self.model = model
        self.optimizer = optimizer  # Fixed: now uses parameter
        self.ds_name = ds_name      # Fixed: now uses parameter
        self.encoding_type = encoding_type
        self.best_accuracy = False
        self.pd_prompt_table = pd.DataFrame()
        
        # Initialize WandB run
        tags = [
            "Ax_optimization", 
            self.ds_name, 
            self.model, 
            self.optimizer, 
            f"randomize={self.randomize}", 
            f"[{self.initial_prompt}]", 
            f"encoding={self.encoding_type}"
        ]
        
        self.run = wandb.init(
            project="prompt_opt_exp", 
            job_type=self.optimizer, 
            tags=tags, 
            group=self.ds_name, 
            name=self.model
        )
        
        self.run.config.update({
            "model": self.model,
            "dataset": self.ds_name,
            "optimizer": self.optimizer,
            "maxiter": 100,
            "randomize": self.randomize,
            "initial_prompt": self.initial_prompt,
            "encoding_type": self.encoding_type
        })

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)[self.ds_name]
            
        # Load ground truth dataset
        self.gt_dataset = load_dataset(
            self.config['GT_IMAGES_DIRECTORY_PATH'],
            self.config['GT_ANNOTATIONS_DIRECTORY_PATH'],
            self.config['GT_DATA_YAML_PATH']
        ) 
        self.gt_dataset = self.set_one_class(self.gt_dataset)
        self.check_classes(self.gt_dataset)
        self.gt_dict = {
            os.path.splitext(os.path.basename(image_path))[0] + ".jpg": (image_path, annotation)
            for image_path, _, annotation in self.gt_dataset
        }
        
        # Initialize tokenizer conditionally
        self.tokenizer = None
        if self.encoding_type == 'bert':
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Initialize with chosen encoding
        self.input_ids = self.encode_prompt(self.initial_prompt)
        
        # If randomizing, modify values
        if self.randomize:
            self._randomize_tokens()
                
        self.ax_client = AxClient()
        print("Initial prompt:", self.decode_prompt(self.input_ids), self.input_ids)
    
    def _randomize_tokens(self):
        """Randomize input token IDs based on encoding type."""
        if self.encoding_type == 'ascii':
            # Randomize ASCII values
            for i in range(random.randint(1, len(self.input_ids))):
                self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(32, 126)
        else:
            # Randomize BERT token IDs
            for i in range(random.randint(1, len(self.input_ids))):
                self.input_ids[random.randint(0, len(self.input_ids)-1)] = random.randint(0, len(self.tokenizer))

    @staticmethod
    def set_one_class(gt_dataset):
        """Set all classes to a single defect class."""
        for key in gt_dataset.annotations.keys():
            gt_dataset.annotations[key].class_id = np.zeros_like(gt_dataset.annotations[key].class_id)
        gt_dataset.classes = ['defect']
        return gt_dataset

    @staticmethod
    def check_classes(gt_dataset):
        """Verify all annotations have the expected class ID."""
        for key in gt_dataset.annotations.keys():
            for i in range(len(gt_dataset.annotations[key])):
                if gt_dataset.annotations[key][i].class_id != len(gt_dataset.classes) - 1:
                    return False
        return True

    def encode_prompt(self, text):
        """Convert a string to token representation based on encoding type."""
        if self.encoding_type == 'ascii':
            return torch.tensor([ord(c) for c in text], dtype=torch.float32)
        else:
            # BERT tokenization
            input_ids = self.tokenizer(text)['input_ids']
            # Remove special tokens (first and last)
            return torch.tensor(input_ids[1:-1], dtype=torch.float32)

    def decode_prompt(self, token_ids, _=None):
        """Convert token IDs back to a string based on encoding type."""
        if self.encoding_type == 'ascii':
            # Round and clip to valid ASCII range (32-126 for printable chars)
            rounded = torch.round(token_ids).clamp(32, 126).to(torch.int64)
            # Convert ASCII values back to characters
            return ''.join([chr(int(code)) for code in rounded])
        else:
            # BERT decoding
            rounded = torch.round(token_ids).clamp(0, len(self.tokenizer)).to(torch.int64)
            return self.tokenizer.decode(rounded, skip_special_tokens=True)

    def step(self, prompt: str, eval_metrics: bool = False):
        """Run inference with the given prompt and evaluate results."""
        try:
            args = argparse.Namespace(
                config='/zhome/4a/b/137804/Desktop/autolbl/config.json',
                section=self.ds_name,
                model=self.model,
                tag='default',
                sahi=False,
                reload=False,
                ontology=f'{prompt}: defect',
                wandb=False,
                save_images=False
            )
            dataset = run_any_args(args)
            
            if eval_metrics:
                confusion_matrix, acc, _ = evaluate_detections(dataset, self.gt_dataset)
                acc = acc[0]
                print(f"Accuracy: {acc}")
                gt_class = "defect"
                TP = confusion_matrix[0, 0] / confusion_matrix.sum()
                FP = confusion_matrix[0, 1] / confusion_matrix.sum()
                FN = confusion_matrix[1, 0] / confusion_matrix.sum()
                F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            else:
                gt_class = "defect"
                TP = None
                FP = None
                FN = None
                acc = None
                F1 = None
                
            if self.best_accuracy:
                compare_wandb_plot(dataset, self.gt_dataset)
                
            return gt_class, TP, FP, FN, acc, F1, dataset
        except Exception as e:
            print(f"Error in step: {e}")
            # Return default values in case of error
            return "defect", 0, 0, 1, 0, 0, None

    def loss2(self, input_ids: torch.Tensor, eval_metrics: bool = False):
        """Calculate loss for given prompt token IDs."""
        try:
            prompt = self.decode_prompt(input_ids)
            print("Loss is being calculated for prompt:", prompt, "input_ids:", input_ids)
            gt_class, TP, FP, FN, acc, F1, dataset = self.step(prompt, eval_metrics)
            
            if dataset is None:
                return self._handle_empty_dataset(prompt, input_ids, TP, FP, FN, acc, F1)
            
            pred_bboxes = torch.empty(0, 4)
            pred_labels = torch.empty(0)
            gt_bboxes = torch.empty(0, 4)
            gt_labels = torch.empty(0)
            pred_scores = torch.empty(0, 2)
            gt_groups = []
            
            for image_path, _, annotation in dataset:
                name_gt = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
                if name_gt in self.gt_dict:
                    pred_bboxes = torch.cat((pred_bboxes, torch.tensor(annotation.xyxy)))
                    pred_labels = torch.cat((pred_labels, torch.tensor(annotation.class_id)))
                    _, gt_annotation = self.gt_dict[name_gt]
                    gt_bboxes = torch.cat((gt_bboxes, torch.tensor(gt_annotation.xyxy)))
                    gt_labels = torch.cat((gt_labels, torch.tensor(gt_annotation.class_id)))
                    if annotation.confidence is not None:
                        conf_tensor = torch.tensor(annotation.confidence, dtype=torch.float32).flatten()
                        conf_tensor = conf_tensor.unsqueeze(1)
                        score_tensor = torch.cat([1 - conf_tensor, conf_tensor], dim=1)
                        pred_scores = torch.cat((pred_scores, score_tensor), dim=0)

            # Check if we have empty tensors - if so, return a high default loss
            if len(pred_labels) == 0 or len(gt_labels) == 0:
                return self._handle_empty_tensors(prompt, input_ids, TP, FP, FN, acc, F1)
                
            # Original code continues if tensors are not empty
            if len(pred_scores) == 0:
                pred_scores = torch.tensor([[-10, 10]])
                pred_scores = pred_scores.repeat(len(pred_labels), 1).float()
            
            if len(gt_groups) == 0:
                gt_groups = [len(gt_labels)]
                
            pred_labels = F.one_hot(pred_labels.long()).float()
            gt_labels = gt_labels.to(torch.int64)
            num_classes = len(self.gt_dataset.classes) + 1
            
            loss_fn = DETRLoss(nc=num_classes, aux_loss=False, use_fl=False, use_vfl=False)
            batch = {
                'cls': gt_labels.to(torch.int64),
                'bboxes': gt_bboxes.to(torch.float32),
                'gt_groups': gt_groups
            }
            
            pred_bboxes = pred_bboxes.unsqueeze(0).unsqueeze(0).to(torch.float32)
            pred_scores = pred_scores.unsqueeze(0).unsqueeze(0).to(torch.float32)
            
            loss_output = loss_fn.forward(
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                batch=batch
            )
            
            total_loss = 20 * loss_output['loss_class'] + loss_output['loss_bbox'] / 1000 + loss_output['loss_giou']
            
            self._log_metrics(prompt, input_ids, loss_output, total_loss, TP, FP, FN, acc, F1)
            return gt_class, TP, FP, FN, acc, F1, dataset, total_loss, prompt
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return high loss in case of any errors
            return "defect", 0, 0, 1, 0, 0, None, torch.tensor(3000.0), "error"

    def _handle_empty_dataset(self, prompt, input_ids, TP, FP, FN, acc, F1):
        """Handle case where dataset is None."""
        total_loss = torch.tensor(5000.0)  # Very high loss for failed runs
        wandb.log({
            "total loss": total_loss,
            "error": "Dataset creation failed",
            "prompt": prompt
        })
        
        new_row = pd.DataFrame({
            'prompt': [prompt], 
            'TP': [0], 
            'FP': [0], 
            'FN': [1], 
            'acc': [0], 
            'F1': [0], 
            'error': ['Dataset creation failed']
        })
        
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        
        return "defect", 0, 0, 1, 0, 0, None, total_loss, prompt
    
    def _handle_empty_tensors(self, prompt, input_ids, TP, FP, FN, acc, F1):
        """Handle case with empty tensors."""
        total_loss = torch.tensor(3000.0)  # High default loss value
        wandb.log({
            "loss_giou": 1000.0,
            "bbox loss": 1000.0,
            "class loss": 1000.0,
            "total loss": total_loss,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1,
            "prompt": prompt,
            "input_ids": input_ids,
            "error": "Empty tensors detected"
        })
        
        new_row = pd.DataFrame({'prompt': [prompt], 'TP': [TP], 'FP': [FP], 'FN': [FN], 'acc': [acc], 'F1': [F1], 'error': ['Empty tensors']})
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})
        
        return "defect", TP, FP, FN, acc, F1, None, float(total_loss), prompt

    def _log_metrics(self, prompt, input_ids, loss_output, total_loss, TP, FP, FN, acc, F1):
        """Log metrics to wandb."""
        wandb.log({
            "loss_giou": loss_output["loss_giou"],
            "bbox loss": loss_output["loss_bbox"],
            "class loss": loss_output['loss_class'],
            "total loss": total_loss,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "F1": F1,
            "prompt": prompt,
            "input_ids": input_ids
        })

        # Update the prompt table
        new_row = pd.DataFrame({'prompt': [prompt], 'TP': [TP], 'FP': [FP], 'FN': [FN], 'acc': [acc], 'F1': [F1]})
        self.pd_prompt_table = pd.concat([self.pd_prompt_table, new_row], ignore_index=True)
        # Upload the prompt table to wandb
        wandb.log({'prompt_table': wandb.Table(dataframe=self.pd_prompt_table)})

    def objective(self, x_np):
        try:
            # First clear CUDA cache before each evaluation
            torch.cuda.empty_cache()
            
            x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
            
            # Try to calculate loss, catching any errors
            try:
                gt_class, TP, FP, FN, acc, F1, dataset, loss, prompt = self.loss2(
                    input_ids=x_tensor,
                    eval_metrics=True
                )
            except RuntimeError as e:
                print(f"Error in loss calculation: {e}")
                # Return a valid but high loss value instead of letting it fail
                return 5000.0
                
            print(f"Loss: {loss}, Prompt: {prompt}")
            
            # Validate loss value
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss)
                
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"Warning: Invalid loss value {loss_val} detected, using default")
                return 5000.0  # Default high loss
                
            return loss_val
        except Exception as e:
            print(f"Unexpected error in objective: {e}")
            return 5000.0  # Return a valid fallback value

    def optimize(self):
        # Define the length of our prompt
        prompt_length = len(self.input_ids)
        
        # Create parameters for each position in the prompt with appropriate bounds
        parameters = []
        if self.encoding_type == 'ascii':
            bounds = [32, 126]  # ASCII printable characters
        else:
            bounds = [0, len(self.tokenizer)]  # BERT vocabulary
            
        for i in range(prompt_length):
            parameters.append({
                "name": f"char_{i}",
                "type": "range",
                "bounds": bounds,
                "value_type": "int",
            })
        
        # Create experiment with individual parameters for each token position
        self.ax_client.create_experiment(
            name="prompt_optimization",
            parameters=parameters,
            objectives={"loss": ObjectiveProperties(minimize=True)},
        )
        # Create generation strategy
        self.generation_strategy = choose_generation_strategy(
            search_space=self.ax_client.experiment.search_space,
            max_parallelism_override=5,  # Set an appropriate value for parallel execution
        )
        
        # Create a custom experiment with MockJobRunner
        metric = lossForMockJobMetric(name="loss")
        experiment = Experiment(
            name="prompt_optimization",
            search_space=self.ax_client.experiment.search_space,
            # Create proper optimization config with our metric
            optimization_config=OptimizationConfig(
                objective=Objective(metric=metric, minimize=True)
            ),
            runner=MockJobRunner(),
            is_test=True
        )
        
        # Replace the experiment in ax_client
        self.ax_client._experiment = experiment
        
        # Create scheduler with the updated experiment
        scheduler = Scheduler(
            experiment=self.ax_client.experiment,
            generation_strategy=self.generation_strategy,
            options=SchedulerOptions(
                # Add these options for better reliability
                #max_pending_trials=2,
                #total_trials=self.run.config.maxiter
            ),
        )
        
        # Create initial parameter dictionary with each token position
        initial_params = {}
        for i in range(prompt_length):
            initial_params[f"char_{i}"] = int(self.input_ids[i].item())
        
        # Attach the initial trial
        self.ax_client.attach_trial(parameters=initial_params)

        # Complete the initial trial
        self.ax_client.complete_trial(
            trial_index=0,
            raw_data={"loss": self.objective(self.input_ids)}
        )
        # Set up the scheduler to run multiple trials
        scheduler = Scheduler(
            experiment=self.ax_client.experiment,
            generation_strategy=self.ax_client.generation_strategy,
            options=SchedulerOptions(),
        )
        
        # Run trials using the scheduler
        max_trials = self.run.config.maxiter
        scheduler.run_n_trials(max_trials)

    def train_evaluate(self, parameterization):
        # Convert the dictionary of individual character parameters back to a list
        x_np = []
        for i in range(len(parameterization)):
            x_np.append(parameterization[f"char_{i}"])
        
        # Make sure we return a native Python float, not a tensor
        loss_value = self.objective(x_np)
        return {"loss": float(loss_value)}



class MockJob(NamedTuple):
    """Dummy class to represent a job scheduled on `MockJobQueue`."""

    id: int
    parameters: Dict[str, Union[str, float, int, bool]]


global_optimizer = None

class MockJobQueueClient:
    """Dummy class to represent a job queue where the Ax `Scheduler` will
    deploy trial evaluation runs during optimization.
    """

    jobs: Dict[str, MockJob] = {}

    def schedule_job_with_parameters(
        self, parameters: Dict[str, Union[str, float, int, bool]]
    ) -> int:
        """Schedules an evaluation job with given parameters and returns job ID."""
        # Code to actually schedule the job and produce an ID would go here;
        # using timestamp in microseconds as dummy ID for this example.
        job_id = int(time() * 1e6)
        self.jobs[job_id] = MockJob(job_id, parameters)
        return job_id

    def get_job_status(self, job_id: int) -> TrialStatus:
        """ "Get status of the job by a given ID. For simplicity of the example,
        return an Ax `TrialStatus`.
        """
        job = self.jobs[job_id]
        # Instead of randomizing trial status, code to check actual job status
        # would go here.
        if randint(0, 3) > 0:
            return TrialStatus.COMPLETED
        return TrialStatus.RUNNING

    def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
        """Get evaluation results for a given completed job."""
        job = self.jobs[job_id]
        # Use global optimizer instead of creating a new one
        if global_optimizer is not None:
            try:
                loss_value = global_optimizer.objective(list(job.parameters.values()))
                # Ensure we're returning a valid float, not NaN
                if not np.isnan(loss_value) and not np.isinf(loss_value):
                    return {"loss": float(loss_value)}
                else:
                    print(f"Warning: Invalid loss value detected: {loss_value}")
                    return {"loss": 5000.0}  # Default high loss
            except Exception as e:
                print(f"Error evaluating job {job_id}: {e}")
                return {"loss": 5000.0}  # Default high loss
        else:
            print("Warning: No global optimizer available")
            return {"loss": 5000.0}  # Default high loss

MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient()
def get_mock_job_queue_client() -> MockJobQueueClient:
    """Obtain the singleton job queue instance."""
    return MOCK_JOB_QUEUE_CLIENT

class MockJobRunner(Runner):  # Deploys trials to external system.
    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        mock_job_queue = get_mock_job_queue_client()
        job_id = mock_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters
        )
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {"job_id": job_id}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(
                job_id=trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)

        return status_dict
    



class lossForMockJobMetric(Metric):  # Pulls data for trial from external system.
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """Obtains data via fetching it from ` for a given trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            mock_job_queue = get_mock_job_queue_client()

            # Here we leverage the "job_id" metadata created by `MockJobRunner.run`.
            loss_data = mock_job_queue.get_outcome_value_for_completed_job(
                job_id=trial.run_metadata.get("job_id")
            )
            df_dict = {
                "trial_index": trial.index,
                "metric_name": "loss",
                "arm_name": trial.arm.name,
                "mean": loss_data.get("loss"),
                # Can be set to 0.0 if function is known to be noiseless
                # or to an actual value when SEM is known. Setting SEM to
                # `None` results in Ax assuming unknown noise and inferring
                # noise level from data.
                "sem": 0,
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

def make_loss_experiment_with_runner_and_metric() -> Experiment:
    parameters = [
        RangeParameter(
            name="x1",
            parameter_type=ParameterType.FLOAT,
            lower=-5,
            upper=10,
        ),
        RangeParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            lower=0,
            upper=15,
        ),
    ]

    objective = Objective(metric=lossForMockJobMetric(name="loss"), minimize=True)

    return Experiment(
        name="loss_test_experiment",
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=MockJobRunner(),
        is_test=True,  # Marking this experiment as a test experiment.
    )



if __name__ == "__main__":
    args = parse_args()
    
    optimizer = PromptOptimizer(
        config_path=args.config,
        encoding_type=args.encoding_type,
        randomize=args.randomize,
        model=args.model,
        optimizer=args.optimizer,
        ds_name=args.ds_name,
        initial_prompt=args.initial_prompt
    )
    global_optimizer = optimizer
    optimizer.optimize()
    optimizer.ax_client.get_trials_data_frame()
    #get best parameters
    best_parameters, values = optimizer.ax_client.get_best_parameters()
    print(f"Best parameters: {best_parameters}")
    #run evaluate with best parameters
    optimizer.best_accuracy = True
    optimizer.loss2(torch.tensor([best_parameters[f"char_{i}"] for i in range(len(best_parameters))]))
    
    wandb.finish()
    torch.cuda.empty_cache()

