import time
import mlflow
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_CHATBOT    = "drupal_chatbot"
EXPERIMENT_SEO        = "drupal_seo_generation"
EXPERIMENT_SUMMARIZE  = "drupal_summarization"
EXPERIMENT_MODERATION = "drupal_moderation"


def _get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id


def track_chatbot(question, answer, sources, response_time, model,
                  spam_detected=False, spam_reason=None):
    try:
        exp_id = _get_or_create_experiment(EXPERIMENT_CHATBOT)
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("model",         model)
            mlflow.log_param("spam_detected",  spam_detected)
            mlflow.log_metric("response_time", round(response_time, 3))
            mlflow.log_metric("answer_length", len(answer))
            mlflow.log_metric("sources_found", len(sources))
            mlflow.set_tag("question_preview", question[:50])
            if spam_reason:
                mlflow.set_tag("spam_reason", spam_reason)
    except Exception as e:
        print(f"MLflow error: {e}")


def track_seo(node_id, title, meta_title, meta_desc, keywords,
              response_time, cached, model):
    try:
        exp_id = _get_or_create_experiment(EXPERIMENT_SEO)
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("model",   model)
            mlflow.log_param("node_id", node_id)
            mlflow.log_param("cached",  cached)
            mlflow.log_metric("response_time", round(response_time, 3))
    except Exception as e:
        print(f"MLflow error: {e}")


def track_summarization(node_id, original_length, summary_length,
                        response_time, cached):
    try:
        exp_id = _get_or_create_experiment(EXPERIMENT_SUMMARIZE)
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("node_id", node_id)
            mlflow.log_param("cached",  cached)
            mlflow.log_metric("response_time",   round(response_time, 3))
            mlflow.log_metric("original_length", original_length)
            mlflow.log_metric("summary_length",  summary_length)
            if original_length > 0:
                compression = round(1 - (summary_length / original_length), 2)
                mlflow.log_metric("compression_ratio", compression)
    except Exception as e:
        print(f"MLflow error: {e}")


def track_moderation(text, label, score, is_toxic, response_time):
    try:
        exp_id = _get_or_create_experiment(EXPERIMENT_MODERATION)
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("label",    label)
            mlflow.log_param("is_toxic", is_toxic)
            mlflow.log_metric("toxicity_score", round(score, 4))
            mlflow.log_metric("response_time",  round(response_time, 3))
            mlflow.set_tag("verdict", "BLOCKED" if is_toxic else "CLEAN")
    except Exception as e:
        print(f"MLflow error: {e}")


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self.start, 3)
