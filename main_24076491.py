
import tensorflow as tf
import matplotlib.pyplot as plt
from constants import (
    TRAINX,
    TRAINY,
    TESTX,
    TESTY,
    SHAPEOFINPUT,
    UNITSLAYER1,
    UNITSLAYER2,
    RATEOFDROPOUT,
    OUTPUTUNITS, 
    HIDDENACTIVATION,
    OUTPUTACTIVATION,
    EPOCHSAMOUNT,
    BATCHSIZE
)

sgd_train_losses = []
sgd_test_losses = []
adam_train_losses = []
adam_test_losses = []
rmsprop_train_losses = []
rmsprop_test_losses = []
hybrid_train_losses = []
hybrid_test_losses = []

model = tf.keras.Sequential([
    tf.keras.layers.Dense(UNITSLAYER1, activation=HIDDENACTIVATION, input_shape=SHAPEOFINPUT),
    tf.keras.layers.Dense(UNITSLAYER2, activation=HIDDENACTIVATION),
    tf.keras.layers.Dropout(RATEOFDROPOUT),
    tf.keras.layers.Dense(OUTPUTUNITS, activation=OUTPUTACTIVATION)
])

optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Function to evaluate the model on the test set and return the loss
def evaluate_loss(test_dataset):
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in test_dataset:
        predictions = model(x_batch, training=False)
        loss_value = loss_fn(y_batch, predictions)
        total_loss += loss_value.numpy() * x_batch.shape[0]  # Total loss for the batch
        total_samples += x_batch.shape[0]
    return total_loss / total_samples

def hybrid_train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    #Gradient for evry optimizer
    gradients_sgd = tape.gradient(loss, model.trainable_variables)
    gradients_adam = tape.gradient(loss, model.trainable_variables)
    gradients_rmsprop = tape.gradient(loss, model.trainable_variables)
    
    #Gradient average acros optimizers
    avg_gradients = []
    for grad_sgd, grad_adam, grad_rmsprop in zip(gradients_sgd, gradients_adam, gradients_rmsprop):
        avg_grad = (grad_sgd + grad_adam + grad_rmsprop) / 3.0
        avg_gradients.append(avg_grad)
    
    #Avg gradients applied using the optimizers
    optimizer_sgd.apply_gradients(zip(avg_gradients, model.trainable_variables))
    optimizer_adam.apply_gradients(zip(avg_gradients, model.trainable_variables))
    optimizer_rmsprop.apply_gradients(zip(avg_gradients, model.trainable_variables))

    return loss

def train_model(train_dataset, test_dataset, optimizer_name, epochs=EPOCHSAMOUNT):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} ({optimizer_name})")
        total_train_loss = 0.0
        total_samples = 0

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            if optimizer_name == 'SGD':
                loss = sgd_train_step(x_batch, y_batch)
            elif optimizer_name == 'Adam':
                loss = adam_train_step(x_batch, y_batch)
            elif optimizer_name == 'RMSprop':
                loss = rmsprop_train_step(x_batch, y_batch)
            else:
                loss = hybrid_train_step(x_batch, y_batch)

            total_train_loss += loss.numpy() * x_batch.shape[0]
            total_samples += x_batch.shape[0]

        # Calculate mean training loss for the epoch
        mean_train_loss = total_train_loss / total_samples
        mean_test_loss = evaluate_loss(test_dataset)  # Calculate test loss

        if optimizer_name == 'SGD':
            sgd_train_losses.append(mean_train_loss)
            sgd_test_losses.append(mean_test_loss)
        elif optimizer_name == 'Adam':
            adam_train_losses.append(mean_train_loss)
            adam_test_losses.append(mean_test_loss)
        elif optimizer_name == 'RMSprop':
            rmsprop_train_losses.append(mean_train_loss)
            rmsprop_test_losses.append(mean_test_loss)
        else:
            hybrid_train_losses.append(mean_train_loss)
            hybrid_test_losses.append(mean_test_loss)

        print(f"Train Loss: {mean_train_loss:.4f}, Test Loss: {mean_test_loss:.4f}")


def sgd_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_sgd.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def adam_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_adam.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def rmsprop_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_rmsprop.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Prepare the datasets
train_dataset = tf.data.Dataset.from_tensor_slices((TRAINX, TRAINY)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((TESTX, TESTY)).batch(32)

# Train the model using each optimizer and record the losses
train_model(train_dataset, test_dataset, optimizer_name='SGD', epochs=EPOCHSAMOUNT)
train_model(train_dataset, test_dataset, optimizer_name='Adam', epochs=EPOCHSAMOUNT)
train_model(train_dataset, test_dataset, optimizer_name='RMSprop', epochs=EPOCHSAMOUNT)
train_model(train_dataset, test_dataset, optimizer_name='Hybrid', epochs=EPOCHSAMOUNT)

def evaluate_model(test_dataset):
    total_correct = 0
    total_samples = 0
    for x_batch, y_batch in test_dataset:
        predictions = model(x_batch, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)
        true_labels = tf.argmax(y_batch, axis=1)
        total_correct += tf.reduce_sum(tf.cast(predicted_labels == true_labels, dtype=tf.float32))
        total_samples += x_batch.shape[0]
    
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    
def plot_losses(training_losses, testing_losses, title):
    plt.plot(training_losses, label='Training Loss')
    plt.plot(testing_losses, label='Testing Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def calculate_accuracy(test_dataset):
    total_correct = 0
    total_samples = 0
    for x_batch, y_batch in test_dataset:
        predictions = model(x_batch, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)
        true_labels = tf.argmax(y_batch, axis=1)
        total_correct += tf.reduce_sum(tf.cast(predicted_labels == true_labels, dtype=tf.float32))
        total_samples += x_batch.shape[0]
    
    accuracy = total_correct / total_samples
    return accuracy.numpy()

# Prepare the test dataset (ensure test_dataset is created similarly as before)
test_dataset = tf.data.Dataset.from_tensor_slices((TESTX, TESTY)).batch(BATCHSIZE)

# Calculate and print the test accuracy
test_accuracy = calculate_accuracy(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plot_losses(sgd_train_losses, sgd_test_losses, "SGD Optimizer")
plot_losses(adam_train_losses, adam_test_losses, "Adam Optimizer")
plot_losses(rmsprop_train_losses, rmsprop_test_losses, "RMSprop Optimizer")
plot_losses(hybrid_train_losses, hybrid_test_losses, "Hybrid Optimizer")


#Model evaluation
evaluate_model(test_dataset)