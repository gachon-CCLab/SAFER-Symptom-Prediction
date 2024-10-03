import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import pandas as pd
# Setting the device (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Defining sequence columns (features)
seq_cols = ['Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy',
       'Normalized_Eight_Hour_Entropy', 'first_TOTAL_ACCELERATION','Location_Variability',
       'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
       'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION',
       'min_TOTAL_ACCELERATION', 'std_TOTAL_ACCELERATION',
       'nunique_TOTAL_ACCELERATION', 'delta_CALORIES', 'first_HEARTBEAT', 'last_HEARTBEAT',
       'mean_HEARTBEAT', 'median_HEARTBEAT', 'max_HEARTBEAT', 'min_HEARTBEAT',
       'std_HEARTBEAT', 'nunique_HEARTBEAT', 'delta_DISTANCE', 'delta_SLEEP',
       'delta_STEP','sex', 'age', 'place_Unknown', 'place_hallway',
       'place_other', 'place_ward']

# Function to evaluate the model's performance
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions, actuals = [], []
    with torch.no_grad():  # No need to calculate gradients during evaluation
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())  # Collect model predictions
            actuals.extend(targets.cpu().numpy())  # Collect actual values
    predictions = np.array(predictions).squeeze()
    actuals = np.array(actuals).squeeze()
    return mean_squared_error(actuals, predictions)  # Return mean squared error

# Function to compute permutation importance
def permutation_importance(model, data_loader, baseline_score, seq_cols):
    importances = np.zeros(len(seq_cols))  # Initialize importances array

    for i, col in enumerate(seq_cols):
        # Create a copy of the data
        permuted_loader = []
        for inputs, targets in data_loader:
            inputs = inputs.clone().detach()  # Detach the inputs from the computation graph
            permuted_inputs = inputs.cpu().numpy()  # Convert to NumPy array
            np.random.shuffle(permuted_inputs[:, :, i])  # Shuffle the feature values (permuting one feature)
            permuted_inputs = torch.tensor(permuted_inputs, dtype=torch.float32).to(device)  # Convert back to Tensor
            permuted_loader.append((permuted_inputs, targets))
        
        # Evaluate the model on permuted data
        permuted_score = evaluate_model(model, permuted_loader)
        
        # Compute importance as the difference between the permuted and baseline score
        importances[i] = permuted_score - baseline_score
    
    return importances
model = 'define_your_model'
test_loader = 'define_your_test_dateset'
# # Compute the baseline performance score of the model
baseline_score = evaluate_model(model, test_loader)

# Compute permutation importance
importances = permutation_importance(model, test_loader, baseline_score, seq_cols)

# Save feature importance results
feature_importance_df = pd.DataFrame({
    'Feature': seq_cols,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.to_excel('permutation_feature_importance.xlsx', index=False)

print("Permutation feature importances calculated and saved to 'permutation_feature_importance.xlsx'")
