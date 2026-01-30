from sklearn import linear_model
reg = linear_model.LinearRegression()
# Train on [[0, 0], [1, 1], [2, 2]] with targets [0, 1, 2]
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print("Model trained successfully.")
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# Test prediction
test_data = [[3, 3]]
prediction = reg.predict(test_data)
print(f"Prediction for {test_data}: {prediction}")
