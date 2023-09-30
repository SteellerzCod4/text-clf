from scripts.train import my_model
from src.data.data_preprocessing import X_test, y_test

y_pred = my_model.predict(X_test)
print(f"Model accuracy: {(y_pred.numpy() == y_test).sum().item() / y_test.shape[0]}")
