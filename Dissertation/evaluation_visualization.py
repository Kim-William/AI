
from pyexpat import model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import os

class Evaluation_Visualization():
    def __init__(self, out_result_dir="Output/result", out_models_dir="Output/models"):
        # Dictionary to store the results
        self.results = {
            'Model': [],
            'Training-Time':[],
            'Accuracy': [],
            'Precision (Class 0)': [],
            'Precision (Class 1)': [],
            'Recall (Class 0)': [],
            'Recall (Class 1)': [],
            'F1-Score (Class 0)': [],
            'F1-Score (Class 1)': []
        }
        self.OUTPUT_RESULT_DIR = out_result_dir
        self.OUTPUT_MODELS_DIR = out_models_dir
    
    # Define a function to plot training history
    def plot_training_history(self, history, title="Model Training History"):
        # Extract values from history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Plot training and validation accuracy
        plt.figure(figsize=(14, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title(f"{title} - Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title(f"{title} - Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Function to plot training history from defaultdict data
    def plot_training_history_from_dict(self, history, title="Model Training History"):
        # Extract values from the dictionary
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']

        # Set up epoch range
        epochs = range(1, len(acc) + 1)

        # Plot training and validation accuracy
        plt.figure(figsize=(14, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title(f"{title} - Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title(f"{title} - Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Function to calculate accuracy and classification report
    def _evaluate_model(self, training_time, model_name, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred, output_dict=True)

        # Store the results
        self.results['Model'].append(model_name)
        self.results['Training-Time'].append(training_time)
        self.results['Accuracy'].append(accuracy)
        self.results['Precision (Class 0)'].append(report['0']['precision'])
        self.results['Precision (Class 1)'].append(report['1']['precision'])
        self.results['Recall (Class 0)'].append(report['0']['recall'])
        self.results['Recall (Class 1)'].append(report['1']['recall'])
        self.results['F1-Score (Class 0)'].append(report['0']['f1-score'])
        self.results['F1-Score (Class 1)'].append(report['1']['f1-score'])

    def _predict_model(self, model,X):
        y_pred_prob = model.predict(X)
        return [1 if prob > 0.5 else 0 for prob in y_pred_prob]

    def evaluate_model_class(self, model_class, X_test, y_test, file_name = None):
        if file_name is None:
            file_name = f'HyperParameter_tuning_result_{model_class.model_name}.xlsx'
        y_pred = self._predict_model(model_class.model, X_test)
        y_pred_random =self. _predict_model(model_class.random_search_cv.best_estimator_, X_test)
        y_pred_grid = self._predict_model(model_class.grid_search_cv.best_estimator_, X_test)
        y_pred_best = self._predict_model(model_class.best_model, X_test)

        self._evaluate_model(model_class.training_time, model_class.model_name, y_test, y_pred)
        self._evaluate_model(model_class.random_search_time,  model_class.model_name + '_random_search', y_test, y_pred_random)
        self._evaluate_model(model_class.grid_search_time,  model_class.model_name + '_grid_search', y_test, y_pred_grid)
        self._evaluate_model(model_class.best_training_time,  model_class.model_name + '_best', y_test, y_pred_best)

        self.df_results = pd.DataFrame(self.results)
        self.df_results.to_excel(os.path.join(self.OUTPUT_RESULT_DIR, file_name))

    def compare_models_accuracy_and_get_best_params(self, models, X_test, y_test):
        best_accuracy = 0
        best_params = None
        best_model_name = None

        for model_name, model_class in models.items():
            # Get model's parameters (either from random search CV or from original model)
            if model_name!='Original':  # Check if it has random_search_cv
                y_pred_prob = model_class.predict(X_test)
                params = model_class.best_params_
            else:
                y_pred_prob = model_class.predict(X_test)
                params = model_class.get_params()

            # Convert probabilities to binary predictions
            pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

            # Calculate accuracy
            accuracy = accuracy_score(y_test, pred) * 100
            print(f'{model_name} Accuracy: {accuracy}')

            # Compare and keep track of the model with the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_model_name = model_name

        print(f'Best Model: {best_model_name} with Accuracy: {best_accuracy}')
        print(f'Best Parameters: {best_params}')
        return best_model_name, best_params
