import math
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.base import clone

import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class RandomBootstrap(object):
    """This class has a method boostrap than randomly get samples 
       from the unlabeled pool.
    """
    def __init__(self, seed):
        """
        **Parameters**
        :seed: A seed (int) that can be used to control your experiments"""
        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y=None, k=1):
        """
        **Parameters**
        :pool: indexes of elements available in the pool of unlabelled data
        * y - None or possible pool
        * k (*int*) - 1 or possible bootstrap size
        **Returns**
        * [] - the chosen array of indexes"""
        return self.randS.chooseNext(pool, k=k)

class BootstrapFromEach(object):
    """This class is used if your experiment was not bootstraped and 
       extracts examples from all classes trying to balance the first
       set of training examples
    """
    def __init__(self, seed):
        """
        **Parameters**
        :seed: A seed (int) that can be used to control your experiments"""
        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y, k=1):
        """
        **Parameters**
        :pool: indexes of elements available in the pool of unlabelled data
        :y: labels for the given X
        :k: 1 or step size
        **Returns**
        * [] - the chosen array of indexes"""

        """ In data for each class label we have a list of indices belonging to that class.
          Then by counting the unique keys in data we count the number of different dataset present """

        data = defaultdict(lambda: [])
        for i in pool:
            data[y[i]].append(i)
        chosen = []
        num_classes = len(data.keys())
        available_indices = set(pool)  # Set of all available indices
        
        for label in data.keys():
            candidates = data[label] #all data against a label
            #print('CHECK ', candidates)
            #print('checking candidates len ', len(candidates))
            # k is bootstrap_size, num_classes is the # of labels, normalize for each
            #10 taken for a particular label rest 10 for the next label in this way we get step size equivalent data
            indices = self.randS.chooseNext(candidates, k=k/num_classes) #the chosen array of indexes
            #print('INDICES ', indices)
            chosen.extend(indices)
            #print('len of choosen ! ',len(chosen))
            available_indices.difference_update(indices)

#       get the remaining examples from the last index
        if len(chosen) < k:   # aikhane k 150 e pache
            #print('checking k ', k)
            missing_elems = k - len(chosen)
            #print('Final candidate len ', len(candidates))
            #print('Available indices len ', len(available_indices))
            indices = self.randS.chooseNext(available_indices, k=missing_elems)
            #print('Final len of indices ', len(indices))
            chosen.extend(indices) 
            #print('Final len of choosen ! ',len(chosen))           
        return chosen


class BaseStrategy(object):
    """This class is a BaseStrategy of which all pool based active learning
       techniques should inherit from."""
    def __init__(self, seed=0, keep_balanced=False):
        """**Parameters**
        :seed: A seed (int) that can be used to control your experiments"""
        self.randgen = np.random.RandomState(seed)
        self.keep_balanced = keep_balanced
        """ All Active learning strategies differ only on how they select the next batch.
        Therefore, they should reimplement the method below.
        """
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indexes = None, current_train_y = None):
        """@Parameters
          :pool: indexes of elements available in the pool of unlabelled data
          :X: vector with variables values to assemble the model
          :model: A machine learning model from the scikit package
          :y: labels for the given X
          :k: 1 or step size
          :current_train_indexes: None or array of trained indices
          :current_train_y: an array with the possible labels(int) for the training instances
        """
        pass
    
    def _compute_inverted_nexample_frequency(self, current_train_y):
        """ 
        This method is used to swap the class frequencies in the training set.
        Using this method we try to select more examples from the classes which
        the number of examples are low and try to decrease the number of examples 
        selected  from the classes which the number of examples are high. 
        @Parameters
        :current_train_y: all label values used in the current training set
        **Returns**
        [*float*]: the inverted frequency of the class distribution 
        """
#       extract current training data frequency and normalize it (0 - 1)
        freq = np.bincount(current_train_y)
        norm_freq = freq / len(current_train_y)
#       now we sort the frequency to map invert the probabilities
        sorted_freq = np.sort(norm_freq)[::-1] # slicing makes it in decending order
        sorted_indices = np.argsort(np.argsort(norm_freq)) # sorts the input indices in ascending order
#       Now we swap the sorted frequency in a way the class with more
#       examples will receive less examples proportional to the class
#       with the lowest number of examples. In the other way around, 
#       the class with less examples will receive more examples proportional
#       to the class with more examples. 
        inverted_freq = [sorted_freq[i] for i in sorted_indices]
        return inverted_freq
    
    def run_experiment(self, X, y, X_test, y_test, model, total_budget, step_size, seed):
        """@Parameters
          :X: vector with variables values to assemble the model
          :y: labels for the given X (integer values only)
          :X_test: vector with variables values to test the model
          :y_test: labels for the given X_test (integer values only)
          :model: A machine learning model from the scikit package
          :total_budget:  the maximal number of instances to be queried
          :step_size: the increment in training instances by an al strategy and that 
                      tests the performance and save information for each experiment
          :seed: A seed (int) that can be used to control your experiments
          **Returns**
          * [String] - the computed stats for each configuration """    
      # The pool variable is a python set that has the size of the data instances
      # We use y here for simplicity and basically we are storing the indexes
      # That are available in the pool of unlabelled instances
        pool = set(range(len(y)))

      # We will store the indexes of all elements selected for training in the
      # variable below
        train_idx = []
      # bootstrapped starts false so that in the first run we extract some 
      # examples to train the first model
        bootstrapped = False
      # We need to store all distinct labels for balancing purposes when
      # bootstrapping
        labels = np.unique(y)
      
      # Here starts the iterative pipeline of Active Learning
      # The overall idea is to test everytime if the number of selected instances
      # for training reaches the max budget. The other case to stop is when the 
      # pool of unlabelled instances has low number of instances to be selected, 
      # and then the step size cannot select the next round of instances
      
      # We will save all stats in the variable below to collect stats from
      # the execution of the active learning pipeline
        rows_to_save = []
        while len(train_idx) < total_budget and len(pool) >= step_size:
        # We boostrap (select instances from the set) in the first run using
        # the class created below. Note the we control the randomness of everything
        # with the seed            
            if not bootstrapped:
                boot_s = BootstrapFromEach(seed)
                new_idxs = boot_s.bootstrap(pool, y=y, k=int(step_size)) #returns choosen indexes from BootstrapFromEach class
                bootstrapped = True
            else:
                new_idxs = self.chooseNext(pool, X, model, k=int(step_size), current_train_indices=train_idx, current_train_y=y[train_idx])

            # Here we remove the indexes of the new selected instances from the pool
            # of unlabelled instances and add the new indexes to the training data
            pool.difference_update(new_idxs)
            train_idx.extend(new_idxs)

            # Now we assemble the model with the new training data
            # and test its performance in the test data, which is separate 
            # from the pool available for querying
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X_test)
            f_score = f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            row = '{},{},{},{},{},{},{},{}'.format(self.name,self.keep_balanced,model.__class__.__name__,seed,total_budget,len(train_idx),f_score, accuracy)
            #print(row)
            rows_to_save.append(row+str('\n'))
        
        return rows_to_save 


class RandomStrategy(BaseStrategy):
    """This class implements a Random Sampling strategy that is the most common baseline
       to be used when testing active learning strategies. 
       Instead of using an active learning strategy to select the next
       batch to be labeled, you make this selection randomly.
       As can be seen, this class inherits the methods from the BaseStrategy.
    """
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indexes = None, current_train_y = None):
        """Here we override the method BaseStrategy.chooseNext
        @Parameters
        :pool: indexes of elements available in the pool of unlabelled data
        :X: (None) vector with variables values to assemble the model
        :model: (None) A machine learning model from the scikit package
        :y: (None) labels for the given X
        :k: 1 or step size
        :current_train_indexes: (None) array with idxs of trained indices
        :current_train_y: (None) an array with labels for the training instances
        @Returns
        * [int] - the chosen array of indexes"""

        list_pool = list(pool) # ai pool a akta particular label ar data gula ache
        rand_indices = self.randgen.permutation(len(pool)) # oi pool len shoman mixed up combinations ashe
        
        return [list_pool[i] for i in rand_indices[:int(k)]] #selecting randomly 

class UncStrategy(BaseStrategy):
    """This class implements the UncertaintySampling strategy, 
       and inherits everything from the BaseStrategy"""
    def __init__(self, seed=0, sub_pool = None, keep_balanced=False):
        """
        @Parameters
        :seed: A seed (int) that can be used to control your experiments
        :sub_pool: None or sub_pool parameter"""
        super(UncStrategy, self).__init__(seed=seed, keep_balanced=keep_balanced)
        self.sub_pool = sub_pool
        self.name = 'UncertaintySampling'

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Here we overide method BaseStrategy.chooseNext to ensure the method
           behaves differently. In this geral implementation we rely on the 
           model trained with the instances in the labeled pool and call the 
           method predict_prob from the scikit learn. 
          @Parameters
          :pool: indexes of elements available in the pool of unlabelled data
          :X: vector with variables values to assemble the model
          :model: A machine learning model from the scikit package
          :k: 1 or step size
          :current_train_indexes: None or array of trained indices
          :current_train_y: an array with labels for the training instances
          **Returns**
          * [int] - the chosen array of indexes"""
        
        #Add some randomness in the indexes to select distinct values         
        if not self.sub_pool: # if None it will enter here
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

#       If the balancement is kept, we can just select the instances
#       that the classifier is more uncertain
        if not self.keep_balanced: # if False it will enter here
            probs = model.predict_proba(X[candidates])
            uncerts =  1. - np.max(probs, axis=1) #  1. - np.max(probs, axis=1)            
#           The code below maps the values indexes to be matched with the list of candidates
            uis = np.argsort(uncerts)[::-1]
#           Here we get k instances only (the k top on the list)
            chosen = [candidates[i] for i in uis[:k]]
            return chosen
        
        else: # if True it will enter here
            inverted_freq = self._compute_inverted_nexample_frequency(current_train_y)
#           Now we calculate the predictions to each example and get the uncertainty
            probs = model.predict_proba(X[candidates])
            uncerts = 1. - np.max(probs, axis=1)
#           Let's store the labels to filter the examples later
            max_labels = np.argmax(probs, axis=1) # indexes of highest probability in each row
#           The list below will receive the chosen instances per class proportional
#           to the inverted distribution of classes.
            chosen = []
#           filter per class predicted
            for c in range(len(inverted_freq)):                
#               Let's store the map of the filtered array to the original candidates
#               and store it in uis
                mask = np.array(c == np.array(max_labels))
                sort_idx = uncerts.argsort() # array of sorted index of uncerts 
                uis = sort_idx[mask[sort_idx]][::-1]  # indices of samples  wth label c         
#                 k is now proportional to the inverted_frequency
                _k = math.floor(inverted_freq[c]*k) # instead of floor it was round
                _chosen = [candidates[i] for i in uis[:_k]] # selecting this much samples from a particular class
                chosen.extend(_chosen)

                 
#           Complete the list of chosen values if the number of k elems was not selected
            if len(chosen) < k:
#               get difference
                diff = set(candidates)
                diff.difference_update(chosen) # gives set of indices not choosen yet
                vals_missing = k - len(chosen)
                diff = list(candidates)
                _elems = [i for i in diff[:vals_missing]]
                chosen.extend(_elems)
                
            return chosen
            
class QBCStrategy(BaseStrategy):
    """This class implements a Query by committee strategy and inherits BaseStrategy"""
    def __init__(self, classifier=LogisticRegression(), classifier_args=None, seed=0, sub_pool = None, num_committee = 10,keep_balanced=False):
        """Instantiate :mod:`al.instance_strategies.QBCStrategy`
        @Parameters
        :classifier: the classifier that will be used to assemble the committee (default: LogisticRegression)
        :classifier_args: the arguments that will be passed to the classifier (default: '')
        :seed: A seed (int) that can be used to control your experiments
        :sub_pool: 
        :num_committee: 10 or an argument a value for the number of models to assemble
                        for the committe voting """
        super(QBCStrategy, self).__init__(seed=seed,keep_balanced=keep_balanced)
        self.sub_pool = sub_pool
        self.num_committee = num_committee
        self.classifier = classifier
        self.classifier_args = classifier_args
        self.name = 'QBC'


    def _vote_entropy(self, sample):
        """ Computes the vote entropy.
        @Parameters
        :sample: 
        @Returns
        :out: (*int*)"""
        votes = defaultdict(lambda: 0.0)
        size = float(len(sample))

        for i in sample:
            votes[i] += 1.0

        out = 0
        for i in votes:
            aux = (float(votes[i]/size))
            out += ((aux*math.log(aux, 2))*-1.)

        return out

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext to reflect the implemention of 
           such a query using the QBC strategy.
        @Parameters
        :pool: indexes of elements available in the pool of unlabelled data
        :X: vector with variables values to assemble the model
        :model: A machine learning model from the scikit package
        :k: 1 or step size
        :current_train_indexes: None or array of trained indices
        :current_train_y: an array with labels for the training instances
        @Returns
        * [int] - the chosen array of indexes"""

        #Add some randomness in the indexes to select distinct values         
        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

#         if ss.issparse(X):
#             if not ss.isspmatrix_csr(X):
#                 X = X.tocsr()

        # Create bags and initialize array with shape (n_committee, candidates)
        comm_predictions = np.zeros(shape=(self.num_committee, len(candidates)),dtype=int)
        for c in range(self.num_committee):
            # Make sure that we have at least one of each label in each bag
            bfe = BootstrapFromEach(seed=c)
            num_labels = len(np.unique(current_train_y))
            initial = bfe.bootstrap(range(len(current_train_indices)), current_train_y, num_labels)
            # Randomly select indexes, with the possibility of repetition to 
            # generate different sets for the committee members
            r_inds = self.randgen.randint(0, len(current_train_indices), size=len(current_train_indices)-num_labels)            
            r_inds = np.hstack((r_inds, np.array(initial)))
            # Assembling the bag(X), bag(y) for training            
            bag = [current_train_indices[i] for i in r_inds]
            bag_y = [current_train_y[i] for i in r_inds]
            new_classifier = clone(self.classifier)
            new_classifier.fit(X[bag], bag_y)
            # Predicting and storing the outcomes for the current committee member
            predictions = new_classifier.predict(X[candidates])
#             comm_predictions.append(predictions)            
            comm_predictions[c] = predictions

        # Compute disagreement for com_predictions
        disagreements = []
        comm_y_predicted = []
        for i in range(len(comm_predictions[0])):
#           Select num_committee values for each instance and calculate
#           vote entropy
            aux_candidate_predicts = comm_predictions[:,i]
            disagreement = self._vote_entropy(aux_candidate_predicts)
            counts = np.bincount(aux_candidate_predicts)
            most_frequent_value = np.argmax(counts)
            comm_y_predicted.append(most_frequent_value) 
            disagreements.append(disagreement)
        
        if not self.keep_balanced: 
            dis = np.argsort(disagreements)[::-1]
            chosen = [candidates[i] for i in dis[:k]]

        else: 
            inverted_freq = self._compute_inverted_nexample_frequency(current_train_y)
            chosen = []
            for c in range(len(inverted_freq)):                
#               Let's store the map of the filtered array by class to the original 
#               candidates and store it in dis
                mask = np.array(c == np.array(comm_y_predicted))
                sort_idx = np.array(disagreements).argsort()
                dis = sort_idx[mask[sort_idx]][::-1]           
#                 k is now proportional to the class_inverted_frequency
                _k = math.floor(inverted_freq[c]*k) # round chilo in place of floor
                _chosen = [candidates[i] for i in dis[:_k]] 
                chosen.extend(_chosen)
#           Complete the list of chosen values if the number of k elems was not selected
            if len(chosen) < k:
#               get difference
                diff = set(candidates)
                diff.difference_update(chosen)
                vals_missing = k - len(chosen)
                diff = list(candidates)
                _elems = [i for i in diff[:vals_missing]]
                chosen.extend(_elems)
            
        return chosen
    


    