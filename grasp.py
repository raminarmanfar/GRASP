#!/usr/bin/python
import sys, os, getopt, math, copy, glob
import random, operator

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

class IBkClass:
	def __init__(self, knnDegree, distanceMethod='euclidean'):
		self.knnDegree = knnDegree
		self.distanceMethod = distanceMethod

	def kNN(self, selectedTopFeatures, trainingSet, testSet):
		'''
		print selectedTopFeatures
		print '-------------------------TrainSet-----------------------------------'
		print trainingSet
		print '----------------------------testSet--------------------------------'
		print testSet
		print '------------------------------------------------------------'
		'''
		predictions = []
		for x in range(len(testSet)):
			neighbors = self.getNeighbors(trainingSet, testSet[x], self.knnDegree)
			result = self.getResponse(neighbors)
			predictions.append(result)
			### >>> print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		accuracy = self.getAccuracy(testSet, predictions)
		return accuracy

	def euclideanDistance(self, instance1, instance2, length):
		distance = 0
		for x in range(length):
			distance += pow((float(instance1[x]) - float(instance2[x])), 2)
		return math.sqrt(distance)

	def getNeighbors(self, trainingSet, testInstance, k):
		distances = []
		length = len(testInstance)-1
		for x in range(len(trainingSet)):
			dist = self.euclideanDistance(testInstance, trainingSet[x], length)
			distances.append((trainingSet[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	def getAccuracy(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(testSet))) * 100.0

# --- main body of program ---
def getColumn(matrix, i):
	return [row[i] for row in matrix]

def main(argv):
	inputFile = outputFile = filterAlg = localAlg = ''
	testpercent = 0.0
	if len(argv) == 1 and (argv[0] == "--help" or argv[0] == "-h"):
		print 'GRASP HELP...\ngrasp.py [-i <input file> -f <filter alg> -l <local search alg> -t <percentage> [-o <output file>]] | [-h]\n--- OR ---\ngrasp.py [-infile <input file> --filter <filter alg> --local <local search alg> --testpercent <percentage> [--outfile <output file>]] | [--help]'
		print 'Parameters descritions...\n>>> Enter the data file name or all for all files in the folder \n>>> filters:\n\t1) entropy\n\t2) gain (Information Gain)\n\t3) correlation\n>>> local search alg:\n\t1) API (Adjacent Pairwise Interchange)\n\t2) GPI (General Pairwise Interchange)\n\t3) TPU (Top Priority Upfront)\n\t4) VNG (Variable Neighborhood Generation)'
		print '>>> Important notes:\n\ta) Either numbers or names for values of parameters are accepted.\n\tb) order of parameters is not mandatory.\n\tc) Both upper & lower case are accepted.\n>>> Train percent: (should be a real number less than 50.00)'
		#sys.exit()
	else:
		try:
			opts, args = getopt.getopt(argv,"i:o:f:l:t:k:",["infile=", "outfile=", "filter=", "local=", "testpercent=", "knndgee="])
		except getopt.GetoptError:
			print '>>>Error: WRONG parameters passed to the program !!!'
			print 'Please enter your parameters using following format...'
			print 'grasp.py [-i <input file> -f <filter alg> -l <local search alg> -t <percentage> -k <percentage> [-o <output file>]] | [-h]'
			print '--- OR ---'
			print 'grasp.py [-infile <input file> --filter <filter alg> --local <local search alg> --testpercent <percentage> --knndegree <kNN-Degree> [--outfile <output file>]] | [--help]'
			sys.exit(2)

		for opt, arg in opts:
			if opt in ("-i", "--infile"):
				inputFile = arg
			elif opt in ("-o", "--ofile"):
				outputFile = arg
			elif opt in ("-l", "--local"):
				localAlg = arg
			elif opt in ("-f", "--filter"):
				filterAlg = arg
			elif opt in ("-t", "--testpercent"):
				testPercent = float(arg)
			elif opt in ("-k", "--knndgee"):
				knnDegree = int(arg)
		if localAlg.lower() not in ["1", "2", "3", "4", "api", "gpi", "tpu", "vng"]:
			print '>>> error in local search algorithm parameter...\n>>> local search can be one of following items or their corresponding number:\n1) API (Adjacent Pairwise Interchange)\n2) GPI (General Pairwise Interchange)\n3) TPU (Top Priority Upfront)\n4) VNG (Variable Neighborhood Generation)'
			sys.exit(2)
		elif filterAlg.lower() not in ["1", "2", "3", "entropy", "gain", "correlation"]:
			print '>>> error in filter algorithm parameter...\n>>> filter can be one of following items or their corresponding number:\n1) entropy\n2) gain\n3) correlation'
			sys.exit(2)
		elif testPercent >= 50:
			print '>>> error in train percent parameter...\n>>> train percentage should be a real number less than 50.00'
			sys.exit(2)


		# read *.csv files in the current folder
		full_path = os.path.realpath(__file__)
		path, file = os.path.split(full_path)
		filesList = glob.glob(path + "/*.csv")
		print("This file directory and name...")
		print(path + '/ --> ' + file + "\n")

		RCL = 0
		if inputFile.lower() == 'all':
			fileInfo = []
			for fn in filesList:
				path, file = os.path.split(fn)
				getSubsetsCount(file)
				fileInfo.append([path, file, RCL, knnDegree])

			for fn in filesList:
				f = fileInfo.pop(0)
				print(f[0] + '/ --> ' + f[1], 'Operation has been started...')
				ourGRASP(f[1], outputFile, testPercent, filterAlg, localAlg, f[2], f[3])
				print '--> Operation has been finished successfully for the data: (' + bcolors.WARNING + f[1] + bcolors.ENDC + ')...'
		else:
			getSubsetsCount(inputFile)
			print(path + '/ --> ' + inputFile, 'Operation has been started...')
			ourGRASP(inputFile, outputFile, testPercent, filterAlg, localAlg, RCL, knnDegree)
			print '--> Operation has been finished successfully for the data: (' + bcolors.WARNING + inputFile + bcolors.ENDC + ')...'

def getSubsetsCount(inputFile):
	f = open(inputFile, 'r')
	data = f.read()
	data = data.splitlines()
	TotalRecords = len(data) - 1;

	for index in range(len(data)):
		data[index] = data[index].split(',')
	'''
	RCL = len(data[0])
	while RCL >= len(data[0]) or KnnDegree > 100:
		RCL = input("Data: " + inputFile + ", Enter RCL (1.." + str(len(data[0])-1) + "): ")
		KnnDegree = input("Data: " + inputFile + ", Enter Knn-Degree (1..100): ")
	return RCL, KnnDegree
	'''

def ourGRASP(inputFile, outputFile, testPercent, filterAlg, localAlg, RCL, knnDegree=3):
	f = open(inputFile, 'r')
	data = f.read()
	data = data.splitlines()
	TotalRecords = len(data) - 1;

	for index in range(len(data)):
		data[index] = data[index].split(',')

	NumberOfTestSet = (int)((testPercent / 100) * TotalRecords);
	NumberOfTrainingSet = TotalRecords - NumberOfTestSet;

	filterAlgName = ''
	if filterAlg.lower() in ('1', 'entropy'):
		filterAlgName = 'Entropy'
	elif filterAlg.lower() in ('2', 'gain'):
		filterAlgName = 'Information Gain'
	elif filterAlg.lower() in ('3', 'correlation'):
		filterAlgName = 'Correlation'

	localAlgName = ''
	if localAlg.lower() in ('1', 'api'): localAlgName = 'Adjacent Pairwise Interchange'
	elif localAlg.lower() in ('2', 'gpi'): localAlgName = 'General Pairwise Interchange'
	elif localAlg.lower() in ('3', 'tpu'): localAlgName = 'Top Priority Upfront'
	elif localAlg.lower() in ('4', 'vng'): localAlgName = 'Variable Neighborhood Generation'

	print '        Input file:', bcolors.WARNING + inputFile.lower() + bcolors.ENDC
	print '       Output file:', bcolors.WARNING + outputFile.lower() + bcolors.ENDC
	print '        Filter Alg:', bcolors.WARNING + filterAlgName + bcolors.ENDC
	print '         Local Alg:', bcolors.WARNING + localAlgName + bcolors.ENDC
	print 'Number of features:', bcolors.WARNING + str(len(data[0]) - 1) + ' + 1 class' + bcolors.ENDC
	print '          Test set:', bcolors.WARNING + str(testPercent) + '% out of ' + str(TotalRecords) + ' records' + bcolors.ENDC
	print ' Number of records:', bcolors.WARNING + str(TotalRecords) + bcolors.ENDC + ', Number of test set: ' + bcolors.WARNING + str(NumberOfTestSet) + bcolors.ENDC + ', Number of training set: ' + bcolors.WARNING + str(NumberOfTrainingSet) + bcolors.ENDC
	print '               RCL:', bcolors.WARNING + str(RCL) + bcolors.ENDC
	print '        Knn-Degree:', bcolors.WARNING + str(knnDegree) + bcolors.ENDC
	print '          Features:', bcolors.WARNING + str(data[0][0:len(data[0]) - 1]) + bcolors.ENDC
	print '       Class Title:', bcolors.WARNING + str(data[0][len(data[0]) - 1]) + bcolors.ENDC
	classValues = getColumn(data, len(data[0]) - 1)
	classValues.pop(0)
	print '      Class Values:', bcolors.WARNING + str(list(set(classValues))) + bcolors.ENDC + ', # of value(s) in class: ' + bcolors.WARNING + str(len(set(classValues))) + bcolors.ENDC

	categorizedList = []
	for i in range(len(data[0])):
		col = getColumn(data, i) # get i'th column of data
		col.pop(0)
		categorizedList.append([[x, col.count(x)] for x in set(col)])

	classCol = getColumn(data, len(data[0]) - 1);
	classCol.pop(0)
	categorizedListByClass = []
	for i in range(len(data[0])):
		col = getColumn(data, i) # get i'th column of data
		col.pop(0)
		g = zip(col, classCol)
		categorizedListByClass.append([[y[0], y[1], g.count(y)] for y in set(g)])
		#$print 'cate by class: ', categorizedListByClass[i]
	
	initialFeatures = DoFilters(filterAlg.lower(), data, categorizedList, categorizedListByClass, classCol, TotalRecords, RCL, knnDegree, NumberOfTestSet)
	#print initialFeatures

def DoFilters(filterType, data, categorizedList, categorizedListByClass, classCol, TotalRecords, RCL, knnDegree, NumberOfTestSet):
	# --- avg[row, featurename, selected, entropy, gain]
	selectedTopFeatures = []
	selectedFeaturesData = []
	if filterType in ('1', 'entropy'): # --- entropy
		avgs, EntS = doEntropy(data, categorizedList, categorizedListByClass, TotalRecords, RCL)
		rndIndex = ChooseFeaturesByCol(avgs, 50.00, 3)
		print 'Entropy of features...'
		for i in range(len(avgs)):
			if avgs[i][2] == False:
				print '\t' + str(i+1) + ') ', bcolors.OKGREEN + avgs[i][1] + bcolors.ENDC + ' ==> Entropy = ' + bcolors.WARNING + str(avgs[i][3]) + bcolors.ENDC
			else:
				print '\t' + str(i+1) + ') ', bcolors.OKGREEN + avgs[i][1] + bcolors.ENDC + ' ==> Entropy = ' + bcolors.WARNING + str(avgs[i][3]) + bcolors.OKGREEN + ' -->> Selected' + bcolors.ENDC
		print 'Total entropy =', bcolors.WARNING + str(EntS) + bcolors.ENDC
	elif filterType in ('2', 'gain'): # --- information gain
		avgs, EntS = doEntropy(data, categorizedList, categorizedListByClass, TotalRecords, RCL)
		for i in range(len(avgs)): avgs[i].append(round(EntS - avgs[i][3], 4))
		rndIndex = ChooseFeaturesByCol(avgs, 50.00, 4, True)

		print 'Entropy & Information Gain of features...'
		for i in range(len(avgs)):
			if avgs[i][2] == False:
				print '\t' + str(i+1) + ') ' + bcolors.OKGREEN + avgs[i][1] + bcolors.ENDC + ' ==> Entropy = ' + bcolors.WARNING + str(avgs[i][3]) + bcolors.ENDC + ', Information Gain = '+ bcolors.WARNING + str(avgs[i][4]) + bcolors.ENDC
			else:
				print '\t' + str(i+1) + ') ' + bcolors.OKGREEN + avgs[i][1] + bcolors.ENDC + ' ==> Entropy = ' + bcolors.WARNING + str(avgs[i][3]) + bcolors.ENDC + ', Information Gain = '+ bcolors.WARNING + str(avgs[i][4]) + bcolors.OKGREEN + ' -->> Selected' + bcolors.ENDC
		print 'Total entropy =', bcolors.WARNING + str(EntS) + bcolors.ENDC
	elif filterType in ('3', 'correlation'): # correlation
		print '>>> Correlation, ' + bcolors.WARNING + 'this section will be implemented soon...' + bcolors.ENDC

	# --- make test and train set ---
	testSet = []
	tmp = copy.copy(data)
	tmp.pop(0)
	from random import randint
	for i in range(NumberOfTestSet):
		rnd = randint(0, len(tmp) - 1)
		t = tmp.pop(rnd)
		testSet.append(t)
	trainSet = copy.copy(tmp)

	selectedTrainSet = []
	selectedTestSet = []
	selectedTopFeatures = []
	#avgs.pop(rndIndex)
	ibk = IBkClass(knnDegree)
	print 'IBk Classification Accuracies...'
	for i in range(len(avgs)):
		if i <> rndIndex:
			selectedTopFeatures = [avgs[rndIndex], avgs[i]] 
			selectedTrainSet = getSpecificCols(trainSet, [rndIndex, i])
			selectedTestSet = getSpecificCols(testSet, [rndIndex, i])
			accuracy = ibk.kNN(selectedTopFeatures, selectedTrainSet, selectedTestSet)
			accuracy = round(accuracy, 4)
			print '\tFeatures [' + bcolors.OKGREEN + avgs[rndIndex][1] + ', ' + avgs[i][1] + bcolors.ENDC + '] = ' + bcolors.WARNING +  repr(accuracy) + '%' + bcolors.ENDC

			#print 'top features >>>>>> ',selectedTopFeatures
			#print len(selectedTestSet),'test set >>>>>> ',selectedTestSet
			#print len(selectedTrainSet),'train set >>>>> ',selectedTrainSet

	# here i should perform ibk alg with the selectedTopFeatures and other features in the avgs
	del tmp
	return selectedTopFeatures

#def remove_column(matrix, column):
#    return [row[:column] + row[column+1:] for row in matrix]
def getSpecificCols(list, cols):
	restrictedList = []
	restrictedList = [[l[i] for i in cols] for l in list]
	return restrictedList

def doEntropy(data, categorizedList, categorizedListByClass, TotalRecords, RCL):
	#print categorizedList
	#print '>>>>',categorizedListByClass
	#print '>>>',TotalRecords
	for i in range(len(categorizedList)):
		col = getColumn(categorizedList[i], 0)
		for j in range(len(categorizedListByClass[i])):
			fi = col.index(categorizedListByClass[i][j][0])
			dominator = categorizedListByClass[i][j][2]
			denominator = categorizedList[i][fi][1]
			if i == len(categorizedList) - 1:
				denominator = TotalRecords
			v = float(dominator) / denominator
			categorizedListByClass[i][j].append(denominator)
			categorizedListByClass[i][j].append(round(v, 4))
			val = -v * math.log(v, 2)
			if math.isnan(val): val = 0.0;
			categorizedListByClass[i][j].append(round(val, 4))
	EntS = 0;
	avgs = []
	for i in range(len(categorizedList)):
		avgs.append([i + 1, data[0][i], False, 0.0])
		for j in range(len(categorizedList[i])):
			#print categorizedListByClass[i][j]
			categorizedList[i][j].append(0.0)
			for k in range(len(categorizedListByClass[i])):
				if categorizedListByClass[i][k][0] == categorizedList[i][j][0]:
					categorizedList[i][j][2] += categorizedListByClass[i][k][5] #entropy of each class
			categorizedList[i][j][2] = round(categorizedList[i][j][2], 4)
			avgs[i][3] += categorizedList[i][j][2]
			#print data[0][i],'Entropy >>> ',categorizedList[i][j]
			if i == len(categorizedList) - 1:
				EntS += categorizedList[i][j][2]
		avgs[i][3] /= len(categorizedList[i])
		avgs[i][3] = round(avgs[i][3], 4)
		tmp = avgs
	return avgs, EntS

def ChooseFeaturesByCol(avgs, subFeaturesPercentage, colNo, isSortDescended=False):
	# --- choose RCL amout of features with lowest entropy ---
	tmp = avgs.pop() # remove class column from the list
	from operator import itemgetter, attrgetter, methodcaller
	avgs = sorted(avgs, key = itemgetter(colNo), reverse=isSortDescended)
	numberOfSubFeatures = int(subFeaturesPercentage / 100.00 * len(avgs))
	for i in range(0, numberOfSubFeatures):	avgs[i][2] = True
	avgs = sorted(avgs, key = itemgetter(0))
	avgs.append(tmp) # re push class column to the list

	selectedItemsIndex = [t[0]-1 for t in avgs if t[2] == True]
	from random import randint
	rnd = randint(0, numberOfSubFeatures - 1)
	rndIndex = selectedItemsIndex[rnd]
	return rndIndex

def DoLocalSearch(localAlg, CurrentNeighbour, generatedNeighbours):
	resultBits = []
	if localAlg.lower() in ('1', 'api'):
		for i in range(len(CurrentNeighbour) - 1):
			tmp = copy.copy(CurrentNeighbour)
			tmp[i], tmp[i + 1] = tmp[i + 1], tmp[i]
			if not IsNeighborExist(tmp, resultBits, generatedNeighbours): resultBits.append(tmp)
	elif localAlg.lower() in ('2', 'gpi'):
		for gap in range(1, len(CurrentNeighbour)):
			i = 0
			while i + gap < len(CurrentNeighbour):
				tmp = copy.copy(CurrentNeighbour)
				tmp[i], tmp[i + gap] = tmp[i + gap], tmp[i]
				if not IsNeighborExist(tmp, resultBits, generatedNeighbours): resultBits.append(tmp)
				i = i + 1
	elif localAlg.lower() in ('3', 'tpu'):
		for i in range(1, len(CurrentNeighbour)):
			tmp = copy.copy(CurrentNeighbour)
			tmp[0], tmp[i] = tmp[i], tmp[0]
			if not IsNeighborExist(tmp, resultBits, generatedNeighbours): resultBits.append(tmp)
	elif localAlg.lower() in ('4', 'vng'):
		print '>>> Variable Neighborhood Generation, ' + bcolors.WARNING + 'this section will be implemented soon...' + bcolors.ENDC
	return resultBits

if __name__ == "__main__":
	main(sys.argv[1:])