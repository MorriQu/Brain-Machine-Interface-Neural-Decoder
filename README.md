# Brain–Machine Interface Neural Decoder

This MATLAB project implements and compares multiple neural decoding algorithms for brain–machine interfaces (BMIs). Given spike-train recordings from the motor cortex, the goal is to classify movement direction and estimate continuous hand trajectories.

## Key Features
- **Classification**:
  - Support Vector Machine (**SVM**)
  - Multilayer Perceptron (**MLP**)

- **Trajectory Estimation**:
  - Polynomial Regression (**Polynomial**)
  - k-Nearest Neighbors (**k-NN**)
  - Linear Regression (**LR**)
  - Autoencoder (**AE**)

## Usage
To evaluate the decoding performance of each method, run the following commands in MATLAB:

```matlab
% In MATLAB command window
RMSE = testFunction_for_students_MTb('Autoencoder');
RMSE = testFunction_for_students_MTb('Polynomial');
RMSE = testFunction_for_students_MTb('KNN');
RMSE = testFunction_for_students_MTb('LR');
RMSE = testFunction_for_students_MTb('LRA');
````

## Results

The function returns **Root Mean Squared Error (RMSE)** and displays a plot comparing decoded vs. actual trajectories. You can easily compare the methods by running each one and observing:

* **Prediction accuracy (RMSE)**: Evaluate the precision of hand trajectory estimation.
* **Visualization of estimated vs. true hand trajectories**: Plot shows how closely the decoded path matches the actual movement.
* **Computation time for real-time application feasibility**: Measure how fast each model processes neural data.
