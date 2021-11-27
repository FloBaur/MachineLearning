import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# def calc_price(kilometer, m, t):
#    predPrice = m * kilometer + t
#    return predPrice


dfCars = pd.read_csv("files/autos_prepared.csv")

price = dfCars["price"]
kilometer = dfCars["kilometer"]

model = LinearRegression()

model.fit(dfCars[["kilometer"]], dfCars[["price"]])
minx = min(kilometer)
maxx = max(kilometer)

t = str(model.intercept_)
m = str(model.coef_)

predicted = model.predict([[minx], [maxx]]) #betrachte diesen Bereich für die Prediction
plt.scatter(kilometer, price)
plt.plot([minx, maxx], predicted, color="red")
plt.show()

predPrice = model.predict([[50000]])
#predPrice = calc_price(50000, -0.0879714, 15988.72674)

test = predPrice[0][0]
print('Der vorhergesagte Preis ist: ' + str(round(test,2)) + '€')

