import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'area': [1000, 1500, 2000, 2500, 4000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

x = df[['area', 'bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(x, y)

prediction = model.predict([[2200, 3]])

print('Predicted price:', prediction[0])