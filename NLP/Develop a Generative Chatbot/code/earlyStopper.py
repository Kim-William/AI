import logging

class EarlyStopper:
    # Initialize the class with a parameter: patience
    def __init__(self, patience=1):
        self.patience = patience  # The number of epochs with no improvement after which training will be stopped
        self.counter = 0  # Counter to keep track of the number of epochs with no improvement
        self.min_validation_loss = float('inf')  # Initialize the minimum validation loss as infinity

    # Define a method named early_stop that takes validation_loss as a parameter
    def early_stop(self, validation_loss):
        logging.info("Starting early stop check")  # Log the start of the early stop check

        # If the current validation loss is less than the minimum validation loss
        if validation_loss < self.min_validation_loss:
            logging.info(f"Min validation loss before comparison: {self.min_validation_loss}")  # Log the minimum validation loss before comparison
            self.min_validation_loss = validation_loss  # Update the minimum validation loss
            logging.info(f"Min validation loss after comparison: {self.min_validation_loss}")  # Log the minimum validation loss after comparison
            self.counter = 0  # Reset the counter
        # If the current validation loss is greater than or equal to the minimum validation loss (minus a delta)
        elif validation_loss >= (self.min_validation_loss-0.001):
            logging.info(f"Min validation loss before comparison: {self.min_validation_loss}")
            self.counter += 1  # Increment the counter
            logging.info(f"Counter after iteration: {self.counter}.")
            # If the counter is greater than or equal to the patience
            if self.counter >= self.patience:
                logging.info(f"As the counter has reached the patience value, {self.patience}, the training is stopped to prevent overfitting.")
                return True  # Return True to indicate that early stopping should occur
        # If the current validation loss is less than or equal to the minimum validation loss (plus a delta)
        elif validation_loss <= (self.min_validation_loss+0.001):
            logging.info(f"Min validation loss before comparison: {self.min_validation_loss}")
            self.counter += 1  # Increment the counter
            logging.info(f"Counter after iteration: {self.counter}.")
            # If the counter is greater than or equal to the patience
            if self.counter >= self.patience:
                logging.info(f"Counter when performing early stopping: {self.counter}.")
                logging.info(f"As the counter has reached the patience value, {self.patience}, the training is stopped to prevent overfitting.")
                return True  # Return True to indicate that early stopping should occur
        
        logging.info("Finishing early stop check")  # Log the finish of the early stop check
        return False  # Return False to indicate that early stopping should not occur
