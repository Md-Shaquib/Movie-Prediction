import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('F:\Machine Learning\Movie Prediction\cost_revenue_clean..csv')
data.describe()
X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

regression = LinearRegression()
regression.fit(X, Y)
print(regression.coef_)
print(regression.intercept_)
print(regression.score(X,Y))

plt.figure(figsize = (10,6))
plt.scatter(X,Y,alpha =0.3)
plt.plot(X, regression.predict(X),color ='red', linewidth = 4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budeget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()