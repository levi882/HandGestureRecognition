import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Set device to GPU if available, else CPU
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define EarlyStopping class
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0.0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'best_model.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.last_val = 0
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # Since we want to minimize loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.last_val = val_loss
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.last_val = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.last_val:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


# Define the HandGestureClassifier model for 250x250 grayscale images with 4 classes
class HandGestureClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(HandGestureClassifier, self).__init__()
        # Input is 1x128x128

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x128x128
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 32x64x64

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 64x32x32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128x32x32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 128x16x16

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Output: 256x16x16
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # Output: 256x8x8

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Output: 512x8x8
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)  # Output: 512x4x4

        # Dropout Layers
        self.dropout_conv = nn.Dropout(0.3)
        self.dropout_fc = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)  # Output for x classes

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm and pooling
        x = self.pool1(f.relu(self.bn1(self.conv1(x))))
        x = self.pool2(f.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        x = self.pool3(f.relu(self.bn3(self.conv3(x))))
        x = self.pool4(f.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)

        x = self.pool5(f.relu(self.bn5(self.conv5(x))))
        x = self.dropout_conv(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Fully connected layers with ReLU and BatchNorm
        x = f.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)

        x = f.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        x = self.fc3(x)  # Raw logits for CrossEntropyLoss
        return x


# Training, validation and testing functions
def train(model, train_device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(train_device), target.to(train_device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Accumulate loss and accuracy
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate_test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)  # Compute loss

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_confusion_matrix(model, test_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    print(f"Using device: {comp_device}")

    # Define transforms for 250x250 grayscale images
    transforms_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        transforms.Resize((128, 128)),  # Resize to 250x250
        transforms.RandomRotation(10),  # Data augmentation
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Data augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])

    # Load the custom dataset with data augmentation
    custom_dataset = ImageFolder(root='data/custom', transform=transforms_train)

    # Print class names and indices
    class_names = custom_dataset.classes
    class_to_idx = custom_dataset.class_to_idx
    print(f"Class names: {class_names}")
    print(f"Class to index mapping: {class_to_idx}")

    # Calculate dataset sizes for 70% train, 20% validation, 10% test
    custom_dataset_size = len(custom_dataset)
    custom_train_size = int(0.7 * custom_dataset_size)
    custom_val_size = int(0.2 * custom_dataset_size)
    custom_test_size = custom_dataset_size - custom_train_size - custom_val_size

    print(f"Total dataset size: {custom_dataset_size}")
    print(f"Training set size: {custom_train_size}")
    print(f"Validation set size: {custom_val_size}")
    print(f"Test set size: {custom_test_size}")

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Split the dataset
    _, _, custom_test_dataset = random_split(
        custom_dataset, [custom_train_size, custom_val_size, custom_test_size]
    )

    # Create data loaders with optimized parameters
    batch_size = 128  # Smaller batch size for larger images

    # Determine number of workers based on CPU cores, but not more than 4
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers for data loading")

    custom_test_loader = DataLoader(
        custom_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if comp_device.type == 'cuda' else False,
    )

    criterion = nn.CrossEntropyLoss()

    # Load the best model
    best_model = HandGestureClassifier(num_classes=len(class_names))
    best_model.load_state_dict(torch.load('best_hand_gesture_model.pth'))
    best_model.to(comp_device)

    # Evaluate on test set
    test_loss, test_acc = validate_test(best_model, comp_device, custom_test_loader, criterion)
    print(f"Best Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save the model for inference
    # Move to CPU for compatibility
    best_model_cpu = best_model.to('cpu')
    # Use a sample input with the correct dimensions (1x1x250x250)
    traced_script_module = torch.jit.trace(best_model_cpu, torch.rand(1, 1, 128, 128))
    traced_script_module.save("best_hand_gesture_classifier.pt")

    # Plot confusion matrix
    plot_confusion_matrix(best_model, custom_test_loader, class_names, 'cpu')


# This is the crucial part to fix the multiprocessing error
if __name__ == "__main__":
    # Add this for Windows support
    multiprocessing.freeze_support()
    main()
