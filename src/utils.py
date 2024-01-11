import time

import torch
import time
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def eval(model, eval_loader, criterion):
    total_loss = 0.0
    num_correct = 0
    total_eval = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_eval += labels.size(0)
            num_correct += (predicted == labels).sum().item()
    
        avg_loss = total_loss/len(eval_loader)
        eval_acc = num_correct/total_eval
    
    return avg_loss, eval_acc

def train_lsh(model, train_loader, criterion, optimizer, eval_loader, num_epochs=20):
    """
    Trains the model using LSH (Locality Sensitive Hashing) algorithm.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        criterion: The loss function used for training.
        optimizer: The optimizer used for training.
        eval_loader (DataLoader): The data loader for the evaluation dataset.
        num_epochs (int): The number of epochs to train the model (default: 20).

    Returns:
        nn.Module: The trained model.
    """
    train_st = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_iters = 0
        epoch_activations = 0
        epoch_st = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_iters += 1
            epoch_activations += len(model.hidden_fc.last_neurons)
        
        model.rebuild_lsh()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_st

        train_loss, train_acc = eval(model, train_loader, criterion)
        test_loss, test_acc = eval(model, eval_loader, criterion)
        # Print the training loss and accuracy after each epoch
        print(f"Average activations: {model.hidden_fc.avg_activ} " +
              f"| Average Epoch activations: {epoch_activations/train_iters}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% ' +
            f'| Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}')
        print()

    train_end = time.time()
    total_training_time = train_end - train_st
    print(f"Total training time: {total_training_time:.2f} seconds")
    return model

def train(model, train_loader, criterion, optimizer, eval_loader, num_epochs=20):
    train_st = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_iters = 0
        epoch_st = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_iters += 1
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_st

        train_loss, train_acc = eval(model, train_loader, criterion)
        test_loss, test_acc = eval(model, eval_loader, criterion)
        # Print the training loss and accuracy after each epoch
        print(f"Epoch time: {epoch_time:.2f} seconds")
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% ' +
            f'| Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}')
        print()

    train_end = time.time()
    total_training_time = train_end - train_st
    print(f"Total training time: {total_training_time:.2f} seconds")
    return model