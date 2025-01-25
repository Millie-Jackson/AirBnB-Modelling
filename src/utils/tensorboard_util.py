# src/utils/tensorboard_util.py

from torch.utils.tensorboard import SummaryWriter



def get_tensorboard_writer(log_dir='./logs'):
    """Create and return a TensorBoard SummaryWriter."""

    return SummaryWriter(log_dir=log_dir)

def log_training_metrics(writer, epoch, train_loss, validation_loss, train_rmse):
    """
    Log scalar metrics (loss and RMSE) for training and validation.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch.
        train_loss (float): Training loss.
        validation_loss (float): Validation loss.
        train_rmse (float): Training RMSE.
    """

    writer.add_scalars("Loss", {"Train": train_loss, "Validation": validation_loss}, epoch)
    writer.add_scalar("RMSE", train_rmse, epoch)

def log_model_histograms(writer, model, epoch):
    """
    Log weight and bias histograms for model layers.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        model (nn.Module): Neural network model.
        epoch (int): Current epoch.
    """

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

'''
Training/Validation Loss: Log the training and validation losses at each epoch to monitor overfitting or underfitting.
Accuracy: Track training and validation accuracy over epochs to evaluate model performance.
Learning Rate: Visualize how the learning rate changes during training (useful for learning rate schedulers).
Model Weights/Gradients: Inspect histograms of model weights and gradients to ensure they are updating as expected.
Custom Scalars: Log any scalar values that are significant for your specific task (e.g., F1-score, precision, recall, etc.).
Images: Visualize sample input images, predictions, and ground truth to ensure the model behaves as expected.
Model Graph: Add the model graph to TensorBoard to better understand its structure and connections.
Confusion Matrix: Log a confusion matrix to understand the classification performance across all classes.
Custom Metrics: Log task-specific metrics (e.g., IoU for segmentation, BLEU for NLP, etc.).
Hyperparameters: Track hyperparameters and experiment results to compare different runs efficiently.
'''