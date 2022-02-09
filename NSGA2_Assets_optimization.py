# import from system libraries of python
import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
from datetime import datetime

# PYMOO
# import all the modules for the pymoo library
import autograd.numpy as anp
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.petal import Petal

# import this module from yahoofinancials library
from yahoofinancials import YahooFinancials


# It is a simple function that download the datas
# from a single stock and store in a json object
# the frequency is monthly but the same can be done daily
# if more accuracy is required but also huge amount of datas is managed.
# start = start date object
# end = end date object
# ticker = name of the stock (string object)

def retrieve_stock_data(ticker, start, end):
    json = YahooFinancials(ticker).get_historical_price_data(start, end, "monthly")
    df = pd.DataFrame(columns=["open","close","adjclose"])
    for row in json[ticker]["prices"]:
        date = datetime.fromisoformat(row["formatted_date"])
        df.loc[date] = [row["open"], row["close"], row["adjclose"]]
    df.index.name = "date"
    return df

# this function take a list of stocks
# and compute some statistical parameters
# as mean or variance for a gaussian distribution
# it uses the "retrieve_stock_data" function in each iteration,
# there is also a plot of the returns in a 10-years period of time

def get_data_for_stocks(stocks):
	str_vol = {}
	str_returns = {}
	L = len(stocks)
	Mean = np.zeros(shape=(L, L),dtype=np.float32)
	
	for index, stock in enumerate(stocks):
		data = retrieve_stock_data(stock, "2011-01-01", "2021-01-01")
		str_returns[stock], mean, m_returns, a_returns = annual_return(data, stock)
		Mean[:,[index]] = np.resize(a_returns, (L,1))
		tdf, tmean, tsigma = stat.t.fit(m_returns)
		str_vol[stock] = annual_volatility(data, stock)
		
		support = np.linspace(m_returns.min(), m_returns.max(), 100)
		m_returns.hist(bins=40, density=True, histtype="stepfilled", alpha=0.5);
		plt.plot(support, stat.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
		plt.title("Monthly returns on "+stock+", 2011-2021", weight="bold");
		plt.show()
	return str_vol, str_returns, np.cov(Mean)


# simple function that compute annual return from montly
# changes of returns, it separes datas in sublists of 12 months
# and print a mean value

def annual_return(data, name):
	returns = data["adjclose"].pct_change().dropna()
	mean = returns.mean()
	n = 12
	returns_sublists = [returns[i:i+n] for i in range(0, len(returns), n)]
	final = [12*np.mean(s) for s in returns_sublists]
	yearly = str(round(np.array(final).mean(), 4)*100)
	print("Annual Return for "+name+": "+yearly+" %")
	return yearly, mean, returns, final
	
# simple function that compute annual volatility from montly
# volatility or standard deviation, the formula is 
# "annual_std = sqrt(21)*monthly_std"

def annual_volatility(data, name):
	data['Log returns'] = np.log(data['close']/data['close'].shift())
	volatility = data['Log returns'].std()*21**.5
	annual_volatility = str(round(volatility, 4)*100)
	print("Annual volatility for "+name+": "+annual_volatility+" %")
	return annual_volatility

# here our example of 6 stocks (tech stocks)
tech_stocks = ['AAPL','MSFT','INTC','AMZN','NVDA','FB']
#retrieve the datas and also store a covariance matrix in "covariance" object
dict_vol, dict_returns, covariance = get_data_for_stocks(tech_stocks)

# Problem Definition - 2 objective func. and 2 constraints
# the reference for this class is in the Pymoo library:
# https://pymoo.org/problems/definition.html
# in this case a simple "Problem" subclass is used

class AssetsOptim(Problem):

	def __init__(self, len, xa, xb, cov):
		super().__init__(n_var=len, n_obj=2, n_constr=3, xl=np.zeros(len), xu=np.ones(len))
        # store our variables in this class
		self.ret = xb
		self.sigma = xa
		self.cov = cov

	def _evaluate(self, x, out, *args, **kwargs):
		r = self.ret
		s = self.sigma
		c = self.cov

		f2 = -anp.dot(x,r.T)
		f1 = anp.matmul(x,(s.T)**2)+2*(anp.matmul(x,anp.matmul(c, x.T)).sum(axis=1))
		
		g1 = anp.sum(x, axis=1)-1 # sum(x) <= 1 (over columns)
		g2 =  f1 - 0.20          # dev.std <= 20%
		g3 = 0.30 + f2           # return >= 30%
		
		out["F"] = anp.column_stack([f1, f2])
		out["G"] = anp.column_stack([g1, g2, g3])

# Problem Solver

# here we need to convert dictionaries to arrays of real numbers
obj1 = np.divide(np.array(list(dict_vol.values())).astype(float),100)
obj2 = np.divide(np.array(list(dict_returns.values())).astype(float),100)

# problem is an instance of "AssetsOptim" subclass of "Problem"
problem = AssetsOptim(len(tech_stocks), obj1, obj2, covariance)

# definition of our solver algorithm, we are using NSGA-2 with a population
# of 100 items and only some are optima solutions, the trade-off is visible
# eliminate_duplicate is a built-in function of the algorithm that merge
# the parent and offspring population
algorithm = NSGA2(pop_size=100, n_offsprings=25 ,  # ~ O(2500) computational cost
                    sampling=get_sampling("real_random"),
                    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                    mutation=get_mutation("real_pm", eta=15),
                    eliminate_duplicates=True)
res = minimize(problem, algorithm, ('n_gen', 500),# 500 iterations
				return_least_infeasible=True, # ~ O(50000) new computational cost
               seed=1,save_history=True,verbose=True) 

# desired weigths are stored in res.X
# desired volatility of the portfolio in res.F[0] and desired return in res.F[1]
# lets do a scatter plot of the solutions. (res.opt is the best results' population)

weights = res.opt.get("X")
plot = Scatter(labels=["Volatility", "Return"], title="Annual volatility versus Annual Return")
plot.add(np.abs(res.opt.get("F")), facecolor="none", edgecolor="red")
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.show()

# Now just plot the weights, in case one of the optima solution


plot = Petal(bounds=[0, 1], cmap="tab20",
             labels=[f"w{i}" for i in range(len(weights[0]))],
             title=("Solution W [ 0 ] (ideal)", {'pad': 20}))
plot.add(weights[0])
plot.show()