import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Lab 1\data.csv', sep=';')
x = df['x'].to_numpy()
y = df['y'].to_numpy()
SNR = df['SNR'].to_numpy()

## Helper functions

def compute_distance(x, y):
    # Compute the Euclidean distance from the origin (0,0)
    return np.sqrt(x**2 + y**2)

def fit(distance, SNR, degree):
    # Fit a polynomial regression model of the specified degree
    return np.polyfit(distance, SNR, deg=degree)

def predict(distance, poly_coeffs):
    # Predict the SNR from a given model using polynomial coefficients
    return np.polyval(poly_coeffs, distance)

def evaluate(distance, SNR, poly_coeffs):
    # Compute the Mean Squared Error (MSE) of the polynomial fit on the chosen data
    predictions = predict(distance, poly_coeffs)
    mse = np.mean((predictions - SNR) ** 2)
    return mse

def separate_test(distance, SNR, test_points):
    # Randomly select training and test sets
    indices = np.arange(len(SNR))
    np.random.shuffle(indices)
    test_indices = indices[:test_points]
    train_indices = indices[test_points:]
    
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    return x_train, y_train, x_test, y_test

# Compute distances
distance = compute_distance(x, y)

# Separate the training and test sets
test_points = 20  # Define how many points to use for testing
x_train, y_train, x_test, y_test = separate_test(distance, SNR, test_points)

# Function to perform evaluation with Tikhonov regularization
def evaluate_tikhonov(x_train: np.ndarray, y_train: np.ndarray, lambda_par: float, max_degree: int) -> tuple[tuple, tuple]:
    distance_train = compute_distance(x_train, y_train)
    results = []

    for degree in range(1, max_degree + 1):
        # Fit polynomial model
        poly_coeffs = fit(distance_train, y_train, degree)
        
        # Calculate the Mean Squared Error (MSE)
        mse = evaluate(distance_train, y_train, poly_coeffs)
        
        # Calculate the regularization term
        regularization = lambda_par * np.sum(poly_coeffs ** 2)
        
        # Total loss
        total_loss = mse + regularization
        results.append(total_loss)

    # Best model based on minimum total loss
    best_degree = np.argmin(results) + 1
    best_model = fit(distance_train, y_train, best_degree)

    return best_model, results

# Run the training with Tikhonov regularization
lambda_par = 1  # Regularization parameter
max_degree = 6   # Maximum polynomial degree

# Evaluate with Tikhonov regularization
best_model_tikhonov, results_tikhonov = evaluate_tikhonov(x_train, y_train, lambda_par, max_degree)

# Get the test performance of the best Tikhonov model
distance_test = compute_distance(x_test, y_test)
mse_test_tikhonov = evaluate(distance_test, y_test, best_model_tikhonov)

# Plot the loss as a function of the degree for Tikhonov
plt.plot(range(1, max_degree + 1), results_tikhonov, marker='o', label='Tikhonov Loss')
plt.title('Tikhonov Regularization Loss')
plt.xlabel('Polynomial Degree')
plt.ylabel('Total Loss (MSE + Regularization)')
plt.xticks(range(1, max_degree + 1))
plt.grid(True)
plt.legend()
plt.show()

# Function to perform evaluation with MDL regularization
def evaluate_representation(x_train: np.ndarray, y_train: np.ndarray, lambda_par: float, max_degree: int) -> tuple[tuple, tuple]:
    distance_train = compute_distance(x_train, y_train)
    results = []

    for degree in range(1, max_degree + 1):
        # Fit polynomial model
        poly_coeffs = fit(distance_train, y_train, degree)
        
        # Calculate the Mean Squared Error (MSE)
        mse = evaluate(distance_train, y_train, poly_coeffs)
        
        # Calculate the MDL regularization term (O(2^N))
        representation_length = lambda_par * (2 ** degree)
        
        # Total loss
        total_loss = mse + representation_length
        results.append(total_loss)

    # Best model based on minimum total loss
    best_degree = np.argmin(results) + 1
    best_model_mdl = fit(distance_train, y_train, best_degree)

    return best_model_mdl, results

# Run the training with MDL regularization
best_model_mdl, results_mdl = evaluate_representation(x_train, y_train, lambda_par, max_degree)

# Get the test performance of the best MDL model
mse_test_mdl = evaluate(distance_test, y_test, best_model_mdl)

# Plot the loss as a function of the degree for MDL
plt.plot(range(1, max_degree + 1), results_mdl, marker='o', label='MDL Loss')
plt.title('MDL Regularization Loss')
plt.xlabel('Polynomial Degree')
plt.ylabel('Total Loss (MSE + Representation Length)')
plt.xticks(range(1, max_degree + 1))
plt.grid(True)
plt.legend()
plt.show()

# Function to perform K-fold cross validation
def k_fold_cross_validation(x_train: np.ndarray, y_train: np.ndarray, k: int, max_degree: int) -> tuple[tuple, tuple]:
    n = len(y_train)
    indices = np.arange(n)
    np.random.shuffle(indices)  # Shuffle indices for randomness
    
    fold_size = n // k
    results = []

    for degree in range(1, max_degree + 1):
        fold_errors = []
        
        for fold in range(k):
            # Split the data into training and validation sets
            val_indices = indices[fold * fold_size: (fold + 1) * fold_size]
            train_indices = np.concatenate((indices[:fold * fold_size], indices[(fold + 1) * fold_size:]))
            
            x_train_fold = x_train[train_indices]
            y_train_fold = y_train[train_indices]
            x_val_fold = x_train[val_indices]
            y_val_fold = y_train[val_indices]

            # Compute distances for training and validation sets
            distance_train = compute_distance(x_train_fold, y_train_fold)
            distance_val = compute_distance(x_val_fold, y_val_fold)

            # Fit polynomial model
            poly_coeffs = fit(distance_train, y_train_fold, degree)
            # Evaluate the model on the validation set
            mse = evaluate(distance_val, y_val_fold, poly_coeffs)
            fold_errors.append(mse)

        # Average error over all folds for this degree
        results.append(np.mean(fold_errors))

    # Best model based on minimum validation error
    best_degree = np.argmin(results) + 1
    best_model_kcross = fit(distance, SNR, best_degree)
    
    return best_model_kcross, results

# Run the training with K-fold cross-validation
k = 4  # Number of folds
x_train, y_train, x_test, y_test = separate_test(distance, SNR, test_points)

# Perform K-fold cross-validation
best_model_kcross, results_kcross = k_fold_cross_validation(x_train, y_train, k, max_degree)

# Get the test performance of the best K-cross model
mse_test_kcross = evaluate(distance_test, y_test, best_model_kcross)

# Plot the validation score as a function of the degree
plt.plot(range(1, max_degree + 1), results_kcross, marker='o', label='K-Cross Validation Loss')
plt.title('K-Fold Cross-Validation Scores')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.xticks(range(1, max_degree + 1))
plt.grid(True)
plt.legend()
plt.show()

# Compare performance of the three solutions (Tikhonov, MDL, and K-cross)
# Print test MSE for all models
print(f'Tikhonov Test Mean Squared Error: {mse_test_tikhonov}')
print(f'MDL Test Mean Squared Error: {mse_test_mdl}')
print(f'K-Cross Validation Test Mean Squared Error: {mse_test_kcross}')

# Plot the results for comparison
plt.bar(['Tikhonov', 'MDL', 'K-Cross Validation'], 
        [mse_test_tikhonov, mse_test_mdl, mse_test_kcross], 
        color=['blue', 'orange', 'green'])
plt.title('Test Mean Squared Error for Different Regularizations')
plt.ylabel('Mean Squared Error')
plt.show()
