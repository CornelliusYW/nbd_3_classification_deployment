import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log model training and evaluation results
def log_model_evaluation(accuracy):
    if accuracy:
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")
    else:
        logger.warning("Model accuracy not sufficient. Retraining might be required.")
