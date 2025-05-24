import hydra
import transformers
from omegaconf import OmegaConf

transformers.logging.set_verbosity_info()
pretrain_tasks = __import__("pretrain_task")
inference_tasks = __import__("inference_task")
evaluate_tasks = __import__("evaluate_task")


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(run_config):
    print(OmegaConf.to_yaml(run_config))
    if run_config.task.endswith("InferenceTask"):
        task = getattr(inference_tasks, run_config.task)(run_config)
    elif run_config.task.endswith("PretrainTask"):
        task = getattr(pretrain_tasks, run_config.task)(run_config)
    elif run_config.task.endswith("EvaluateTask"):
        task = getattr(evaluate_tasks, run_config.task)(run_config)
    else:
        raise ValueError(f"Unknown task: {run_config.task}")
    task.run()


if __name__ == "__main__":
    main()
