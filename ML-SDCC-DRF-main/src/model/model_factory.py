from src.model.classification_model import ClassificationModel
from src.model.regression_model import RegressionModel
from src.utils.config import load_config


# dynamically instantiates the correct machine learning model based on dataset metadata
class ModelFactory:

    @staticmethod
    # retrieves dataset configuration and returns the appropriate classification or regression model instance
    def get_model(task_type, target_column):

        # return the corresponding model instance based on the task type
        if task_type == 'classification':
            return ClassificationModel(target_column=target_column)
        elif task_type == 'regression':
            return RegressionModel(target_column=target_column)
        else:
            raise ValueError(f"Task type '{task_type}' not supported. Use 'classification' or 'regression'.")