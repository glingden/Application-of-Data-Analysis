import operator as op
import itertools as it
import pandas as pan
import numpy as np
import matplotlib.pyplot as plt

'''
	Load the data set and normalize
'''
def loadData(filename) :
	
	# Load the data
	data = pan.read_csv(filename).as_matrix()

	# Normalize the data
	for i in range(len(data.T)) :
		mean = np.mean(data.T[i])
		stdv = np.std(data.T[i])
		for j in range(len(data)) :
			data[j][i] = (data[j][i] - mean) / stdv

	return data


'''
	Find the k closest neighbours
		tSet is the training set
		inst is the current instance
		k    is the number of neighbours
'''
def getNeighbours(tSet, inst, k) :
	
	nn = []											# Initialize nearest neighbours array

	for x in range(len(tSet)):
		dist = np.linalg.norm(inst[3:] - tSet[x][3:])
		if x < k :
			nn.append((dist, tSet[x]))				# Put the first k into the array
		else :
			nn.sort(key = op.itemgetter(0))   		# For the rest, sort the array,
			if nn[k-1][0] > dist :					#   compare with the last one,
				nn[k-1] = (dist, tSet[x])			#   and if closer, update it

	return nn

'''
	Predict the class of the instance depending on the neighbors classes
'''
def regressFromNeighbours(neighbours) :
	
	val = np.array([0.0, 0.0, 0.0])					# Regression predicted value

	for x in range(len(neighbours)) :
		
		if neighbours[x][0] == 0 :					# If one of the distances is 0
			return neighbours[x][1][0:3] 			#   return that concentration
		else :
			val += neighbours[x][1][0:3]			# Otherwise add the values to compute the mean

	return val / len(neighbours)


'''
	Cross validation function
'''
def validate(dataSet, k, leaveOut) :

	predVals = []									# Initialize the array to store the predicted values
	i = 0

	while i <= len(dataSet) :	

		testSet = dataSet[i:i+leaveOut]				# Leave out the corresponding number for the test set
		trainingSet = np.concatenate(				# The rest go to the training set
				[dataSet[0:i], 
				dataSet[i+leaveOut:]]
			)

		for x in range(leaveOut) :					# For each element in the test set
			
			if i + x >= len(dataSet) :
				break

			neighbours = getNeighbours(				#	get the neighbours
							trainingSet, 
							testSet[x], 
							k
						)
			predVals.append(						#	and add to the predicted values
						regressFromNeighbours(		#	the mean concentrations
							neighbours
						)
					)

		i += leaveOut

	return np.array(predVals)

'''
	Function to compute the c-index of the predicted values
'''
def cIndex(real, pred) :

	n = np.array([0.0, 0.0, 0.0])
	h_sum = np.array([0.0, 0.0, 0.0])
	for i in range(len(real[0])) :
		for c in range(3) :
			t = real[c][i]
			p = pred[c][i]
			for j in range(i+1, len(real[0])) :
				tt = real[c][j]
				pp = pred[c][j]
				if t != tt :
					n[c] += +1
					if (p < pp and t < tt) or (p > pp and t > tt) :
						h_sum[c] = h_sum[c] + 1
					elif p == pp :
						h_sum[c] = h_sum[c] + 0.5

	return np.divide(h_sum,n)

'''
	Main function to run the previous ones
'''
def main():

	data = loadData('Water_data.csv')				# Load the data

	ks = [1,2,3,4,5,6,8,10,15,20,30]				# Array of k's to test

	realVals = data.T[0:3]							# Array of real values

	cInd1 = np.zeros([3,len(ks)])					# Initialize array for c-index
	cInd3 = np.zeros([3,len(ks)])					# Initialize array for c-index

	for k in range(len(ks)) :

		vals1 = validate(data, ks[k], 1).T[0:3]		# Leave one out cross-validation
		vals3 = validate(data, ks[k], 4).T[0:3]		# Leave one out cross-validation

		cInd1.T[k] = cIndex(realVals, vals1)		# Calculate 1-leave out C-index
		cInd3.T[k] = cIndex(realVals, vals3)		# Calculate 3-leave out C-index

	# Plot the results
	plt.figure(1)
	
	plt.subplot(211)
	plt.title("1-leave out cross-validation")
	plt.plot(ks, cInd1[0], label="Total")
	plt.plot(ks, cInd1[1], label="Cd")
	plt.plot(ks, cInd1[2], label="Pb")
	plt.legend()
	plt.xlabel("K")
	plt.ylabel("C-index")

	plt.subplot(212)
	plt.title("4-leave out cross-validation")
	plt.plot(ks, cInd3[0], label="Total")
	plt.plot(ks, cInd3[1], label="Cd")
	plt.plot(ks, cInd3[2], label="Pb")
	plt.legend()
	plt.xlabel("K")
	plt.ylabel("C-index")
	
	plt.savefig('graphic.png')
	plt.show()


	
# Run main function
main()
