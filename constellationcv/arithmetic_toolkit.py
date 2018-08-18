from __future__ import division
import math
import random
class Vectors:
	def vector_add(v,w):
		"""adds corresponding elements"""
		return [v_i + w_i for v_i, w_i in zip(v,w)]

	def vector_subtract(self,v,w):
		"""subtracts corresponding elements"""
		return [v_i - w_i for v_i,w_i in zip(v,w)]

	def vector_sum(vectors):
		"""sums all corresponding elements"""
		result = vectors[0]
		for vector in vectors[1:]:
			result = vector_add(result, vector)
		return result

	def scalar_multiply(c,v):
		"""c is a number, v is a vector"""
		return [c*v_i for v_i in v]

	def vector_mean(vectors):
		"""compute the vector whose ith element is the mean of the
		ith elements of the input vectors"""
		n = len(vectors)
		return scalar_multiply(1/n,vector_sum(vectors))

	def dot(self,v,w):
		"""v_1 * w_1 + ... v_n * w_n"""
		return sum(v_i * w_i for v_i,w_i in zip(v,w))

	def sum_of_squares(self,v):
		"""v_1 * v_1 + ... + v_n * v_n"""
		return self.dot(v,v)

	def magnitude(self,v):
		"""magnitude(length) of vector v"""
		return math.sqrt(self.sum_of_squares(v))

	def squared_distance(self, v,w):
		"""((v_1 - w_1)^2) + ... ((v_n - w_n)^2)"""
		return self.sum_of_squares(self.vector_subtract(v,w))

	def distance(self, v,w):
		"""distance between vectors 1 and 2"""
		return math.sqrt(self.squared_distance(v,w))

class Matrices:
	def shape(A):
		num_rows = len(A)
		num_cols = len(A[0]) if A else 0
		return num_rows,num_cols

	def get_row(A,i):
		return A[i]

	def get_column(A,j):
		return [A_i[j] for A_i in A]

	def make_matrix(num_rows, num_cols, entry_fn):
		"""return a num_rows x num_cols matrix
		whose (i,j)th entry is entry_fn(i,j)"""
		return [[entry_fn(i,j)
				for  j in range(num_cols)]
				for i in range(num_rows)]

	def is_diagonal(i,j):
		"""1s on the diagonal, 0s everywhere else"""
		return 1 if i==j else 0

class Probability():
	def uniform_pdf(x):
		return 1 if x>=0 and x<1 else 0

	def uniform_cdf(x):
		"""returns the probability that a uniform random variable is <= x"""
		if x<0: return 0
		elif x<1: return x
		else: return 1

	def normal_pdf(x,mu=0,sigma=1):
		sqrt_two_pi = math.sqrt(2*math.pi)
		return (math.exp(-(x-mu)**2/2/sigma**2)/(sqrt_two_pi*sigma))

	def normal_cdf(x,mu=0, sigma=1):
		return (1+math.erf((x-mu)/math.sqrt(2)/sigma))/2

	def inverse_normal_cdf(p,mu=0,sigma=1,tolerance=0.00001):
		"""find approximate inverse using binary search"""
		if mu != 0 or sigma != 1:
			return mu+sigma * inverse_normal_cdf(p,tolerance=tolerance)

		low_z = -10.0
		hi_z = 10.0
		while hi_z - low_z > tolerance:
			mid_z = (low_z+hi_z)/2
			mid_p = normal_cdf(mid_z)
			if mid_p < p:
				low_z = mid_z
			elif mid_p > p:
				hi_z - mid_z
			else:
				break

		return mid_z

	def bernoulli_trial(p):
		return 1 if random.random() < p else 0

	def binomial(n,p):
		return sum(bernoulli_trial(p) for _ in range(n))

class SingleSet:
	def mean(x):
		return sum(x)/len(x)

	def median(v):
		"""finds the 'middle-most' value of v"""
		n = len(v)
		sorted_v = sorted(v)
		midpoint = n//2

		if n%2==1:
			return sorted_v[midpoint]
		else:
			lo = midpoint-1
			hi = midpoint
			return (sorted_v[lo]+sorted_v[hi])/2

	def quantile(x,p):
		"""returns the pth-percentile value in x"""
		p_index = int(p*len(x))
		return sorted(x)[p_index]

	def mode(x):
		"""returns a list, might be more than 1 mode"""
		counts = Counter(x)
		max_count = max(counts.values())
		return [x_i for x_i, count in counts.iteritems()
				if count == max_count]

	def data_range(x):
		return max(x) - min(x)

	def de_mean(x):
		"""translate x by subtracting its mean (so that the result has mean 0)"""
		x_bar = mean(x)
		return [x_i-x_bar for x_i in x]

	def variance(x):
		"""assumes x has at least two elements"""
		n = len(x)
		deviations = de_mean(x)
		return sum_of_squares(deviations)/(n-1)

	def standard_deviation(x):
		return math.sqrt(variance(x))

	def interquartile_range(x):
		return quantile(x,0.75)-quantile(x,0.25)

class Correlation:
	def covatiance(x,y):
		n=len(x)
		return dot(de_mean(x),de_mean(y))/(n-1)

	def correlation(x,y):
		stdev_x = standard_deviation(x)
		stdev_y = standard_deviation(y)
		if stdev_x > 0 and stdev_y > 0:
			return covariance(x,y)/stdev_x/stdev_y
		else:
			return 0