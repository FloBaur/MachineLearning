import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def calc_price(kilometer, m, t):
    predPrice = m * kilometer + t
    return predPrice


dfCars = pd.read_csv("files/autos_prepared.csv")

price = dfCars["price"]
kilometer = dfCars["kilometer"]

model = LinearRegression()

model.fit(dfCars[["kilometer"]], dfCars[["price"]])
miny = min(price)
maxy = max(price)

t = str(model.intercept_)
m = str(model.coef_)

predicted = model.predict([[miny], [maxy]])
plt.scatter(kilometer, price)
plt.plot([0, 50000], predicted, color="red")
plt.show()

predPrice = calc_price(50000, -0.0879714, 15988.72674)

print('Der vorhergesagte Preis ist:')
print(predPrice)
