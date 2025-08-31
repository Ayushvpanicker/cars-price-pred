Car Price Prediction using PyTorch 🚗💰












Overview

Predict the selling price of used cars using a two-layer neural network built in PyTorch. The model captures non-linear relationships between car features and prices, producing realistic, non-negative predictions.

Dataset

Source: Amankharwal Car Dataset

Features: Year, Present_Price, Kms_Driven, Owner, Fuel_Type, Seller_Type, Transmission

Target: Selling_Price

Quick Start
1️⃣ Clone the repository
git clone https://github.com/Ayushvpanicker/cars-price-pred.git
cd cars-price-pred

2️⃣ Install dependencies
pip install torch pandas scikit-learn matplotlib

3️⃣ Launch the Jupyter Notebook
jupyter notebook car_price_prediction.ipynb

Project Workflow
+-------------------+       +---------------------+       +---------------------+
|                   |       |                     |       |                     |
|   Input Features  | ----> |  Neural Network     | ----> |  Predicted Selling  |
|                   |       |  (2-Layer ReLU)    |       |  Price (in lakhs)   |
+-------------------+       +---------------------+       +---------------------+
| Year               |                              
| Present_Price      |                              
| Kms_Driven         |                              
| Owner              |                              
| Fuel_Type (encoded)|                              
| Seller_Type (encoded)|                             
| Transmission (encoded)|                            
+-------------------+                               

Usage
Train Model
history = fit(epochs=300, lr=1e-2, model=model, train_loader=train_loader, val_loader=val_loader)

Evaluate Model
final_result = evaluate(model, val_loader)
print("Validation Loss:", final_result['val_loss'])

Predict Selling Price
example_input = torch.Tensor(scaler_X.transform([[2020, 10.5, 5000, 0]]))
pred_price = max(0, model(example_input).item())
print("Predicted Selling Price:", round(pred_price, 2))

Training Visualization
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

Features

Properly normalized input features

Target kept in real units for realistic predictions

Two-layer neural network captures non-linear relationships

Output clamped to non-negative values

Visualize training and validation losses

Example Result

Prediction for a sample car:

Predicted Selling Price: 14.18 lakhs

License

MIT License © 2025
