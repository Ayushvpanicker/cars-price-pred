

Car Price Prediction using PyTorch 🚗💰






Overview

Predict the selling price of used cars using a two-layer neural network built in PyTorch. The model learns non-linear relationships between car features and prices.

Dataset

Source: Amankharwal Car Dataset

Features: Year, Present_Price, Kms_Driven, Owner, Fuel_Type, Seller_Type, Transmission

Target: Selling_Price

Quick Start
1️⃣ Clone repo
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction

2️⃣ Install dependencies
pip install torch pandas scikit-learn matplotlib

3️⃣ Run Jupyter Notebook
jupyter notebook car_price_prediction.ipynb

Usage Examples
Train Model
history = fit(epochs=300, lr=1e-2, model=model, train_loader=train_loader, val_loader=val_loader)

Evaluate Validation Loss
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

Properly normalized inputs

Realistic, non-negative predictions

Two-layer neural network for non-linear relationships

Easy-to-run notebook with example predictions

Results

Example prediction for a car:

Predicted Selling Price: 14.18 lakhs

License

MIT License © 2025
