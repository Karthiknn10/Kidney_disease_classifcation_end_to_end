import tensorflow as tf
from tensorflow.keras import mixed_precision
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.prepare_base_model import PrepareBaseModel
from CNN_Classifier import logger

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        # Avoid OOM errors by setting GPU Memory Consumption Growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU for training: {tf.config.list_physical_devices('GPU')}")
        mixed_precision.set_global_policy(policy="mixed_float16")


    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

