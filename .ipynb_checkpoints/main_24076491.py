import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf

file_path = 'Almond.csv'  # Update with your file path
almond_df = pd.read_csv(file_path)

if 'Unnamed: 0' in almond_df.columns:
    almond_df = almond_df.drop(columns=['Unnamed: 0'])
    
X = almond_df.drop(columns=['Type'])  # Assuming 'Type' is the target column
y = almond_df['Type']

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

encoder = OneHotEncoder()
y_encoded_sparse = encoder.fit_transform(y.values.reshape(-1, 1))
y_encoded = y_encoded_sparse.toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),  # Input layer with 12 features
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer for 3 classes (Mamra, Sanora, Regular)
])

optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

def hybrid_train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients for each optimizer
    gradients_sgd = tape.gradient(loss, model.trainable_variables)
    gradients_adam = tape.gradient(loss, model.trainable_variables)
    gradients_rmsprop = tape.gradient(loss, model.trainable_variables)
    
    # Average the gradients across optimizers
    avg_gradients = []
    for grad_sgd, grad_adam, grad_rmsprop in zip(gradients_sgd, gradients_adam, gradients_rmsprop):
        avg_grad = (grad_sgd + grad_adam + grad_rmsprop) / 3.0
        avg_gradients.append(avg_grad)
    
    # Apply averaged gradients using the optimizers
    optimizer_sgd.apply_gradients(zip(avg_gradients, model.trainable_variables))
    optimizer_adam.apply_gradients(zip(avg_gradients, model.trainable_variables))
    optimizer_rmsprop.apply_gradients(zip(avg_gradients, model.trainable_variables))

    return loss

def train_model(train_dataset, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss = hybrid_train_step(x_batch, y_batch)
            print(f"Step {step+1}, Loss: {loss.numpy():.4f}")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)  # Batch size of 32

# Train the model for 10 epochs
train_model(train_dataset, epochs=10)

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
    

# Prepare the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Evaluate the model
evaluate_model(test_dataset)