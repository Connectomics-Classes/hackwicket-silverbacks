import matplotlib.pyplot as plt
import numpy as np
import pipeline
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def main():	
	prList = []
	#haven't yet decided on the starting and stopping points and increments
	#probably gonna read a few papers before doing this super long process
	
	#n_segments = range(start, stop, seg) 
	#compactness = range(start,stop,seg)
	#threshold = range(start, stop, seg)
	#nri = range(value)
	
	for a in n_segments:
		for b in compactness:
			for c in threshold:
				for d in nri:
					#returns tuples (precision, recall) and (n_segments, compactness, threshold, nri)
					pr, parameters = pipeline.main(700, .29, .02, 1) 		
					prList.append(pr, parameters)
	
	npArray = np.array(prList)
	
	#finding the parameters for the max precision+recall
	#npArray structure is like this [ [(Precision, Recall),(nseg, compact, thresh, nri)]
	#								[(Precision, Recall),(nseg, compact, thresh, nri)]
	#								..................................................
	#							....[(Precision, Recall),(nseg, compact, thresh, nri)] ]
	
	allPRValues = npArray[:,0,:] #gets all the (P,R) pairs in npArray
	sumPRValues = np.add.reduce(allPRValues, 1) #sums the precision and recall pairs
	index = np.argmax(sumPRValues) #finds the index of the max precision+recall value
	
	maxPR = npArray[index][0]
	maxParameters = npArray[index][1] #parameters that correspond with the max P+R value
	print(maxParameters) #these will be your ideal parameters
	
	
	#Plot Precision-Recall curve
	plt.clf()
	plt.plot(npArray[:,0,0], npArray[:,0,1], label="Precision-Recall curve")
	plt.xlabel("Precision")
	plt.ylabel("Recall")
	plt.xlim([0.0, 1.05])
	plt.ylim([0.0, 1.05])
	plt.title("Precision-Recall Curve")
	plt.legend(loc="lower right")
	plt.show()
	

if __name__ == '__main__':
	main()

