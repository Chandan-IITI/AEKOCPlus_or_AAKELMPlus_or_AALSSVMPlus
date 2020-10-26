# Author: Chandan Gautam
# Institute: IIT Indore, India
# Email: chandangautam31@gmail.com , phd1501101001@iiti.ac.in

# Following papers are implemented in the following codes:
    
# Paper1 (KOC+): Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "KOC+: Kernel ridge regression based one-class classification using privileged information."
#  Information Sciences 504 (2019): 324-333.  

# Paper2 (OCKELM/KOC): Leng, Qian, et al. "One-class classification with extreme learning machine."
#    Mathematical problems in engineering 2015 (2015).

# Paper3 (AEKOC+): Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "AEKOC+: Kernel ridge regression based Auto-Encoder for one-class classification using privileged information."
#    (Submitted after first revision in Cognitive computation, Springer.)

# Paper4 (AEKOC/AAKELM): Gautam, Chandan, Aruna Tiwari, and Qian Leng. "On the construction of extreme learning machine for online and   offline one-class classification-An expanded toolbox."
#   Neurocomputing 261 (2017): 126-143.

# Paper5 (SVDD+): Zhang, Wenbo. "Support vector data description using privileged information."
#    Electronics Letters 51.14 (2015): 1075-1076.
        
# Paper6 (OCSVM+): Zhu, Wenxin, and Ping Zhong. "A new one-class SVM based on hidden information."
#    Knowledge-Based Systems 60 (2014): 35-43. 
# AND
# Burnaev, Evgeny, and Dmitry Smolyakov. "One-class SVM with privileged information and its application to malware detection."
#    2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW). IEEE, 2016.

import numpy as np
from cvxopt import matrix
from cvxopt import sparse
from cvxopt.solvers import qp
import math


def linear_kernel(X, Y=None):
    return np.dot(X, X.T) if Y is None else np.dot(X, Y.T)


class IKOC(object):
    '''
    This class implements KOC+ algorithm,
    
    Paper: Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "KOC+: Kernel ridge regression based one-class classification using privileged information."
    Information Sciences 504 (2019): 324-333.
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.regularization = regularization
        self.privileged_regularization = privileged_regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X, Z):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        Cpriv = self.privileged_regularization
        size = X.shape[0]
        T = np.ones((size, 1))
        self.dual_alpha = np.dot(np.dot(np.linalg.inv(Cpriv * kernel_x + C * (np.dot(kernel_x, kernel_z)) + kernel_z),(np.eye(size)*Cpriv + C * kernel_z)), T)
        self.support_vectors = X
        score1 = abs(T - np.dot(kernel_x,self.dual_alpha))
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        T_one = np.ones((X.shape[0], 1))
        score_test = self.threshold - abs(T_one - np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha)) 
        
        return score_test
    
class KOC(object):
    '''
    This class implements KOC algorithm,
    Paper: Leng, Qian, et al. "One-class classification with extreme learning machine."
    Mathematical problems in engineering 2015 (2015).
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.regularization = regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        size = X.shape[0]
        T = np.ones((size, 1))
        self.dual_alpha = np.dot(np.linalg.inv(np.eye(size)*C + kernel_x), T)
        self.support_vectors = X
        score1 = abs(T - np.dot(kernel_x,self.dual_alpha))
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self
    
    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        T_one = np.ones((X.shape[0], 1))
        score_test = self.threshold - abs(T_one - np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha)) 
        
        return score_test
    
    ### For another threshold criteria
    def fit1(self, X):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        size = X.shape[0]
        T = np.ones((size, 1))
        self.dual_alpha = np.dot(np.linalg.inv(np.eye(size)*C + kernel_x), T)
        self.support_vectors = X
        outY = np.dot(kernel_x,self.dual_alpha)
        self.meanY = np.mean(outY)
        fracrej = self.nu
        self.threshold = self.meanY * fracrej
        self.score_train = abs(outY - self.meanY)
        return self
    
    def decision_function1(self, X):
        """
        Return anomaly score for points in X
        """
        score_test = self.threshold - abs(self.meanY - np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha)) 
        
        return score_test
		
class IAEKOC(object):
    '''
    This class implements AEKOC+ algorithm,
    
    Paper: Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "AEKOC+: Kernel ridge regression based Auto-Encoder for one-class classification using privileged information."
    (Submitted after first revision in Cognitive computation, Springer.)
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.regularization = regularization
        self.privileged_regularization = privileged_regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X, Z):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        Cpriv = self.privileged_regularization
        size = X.shape[0]
        T = X
        self.dual_alpha = np.dot(np.dot(np.linalg.inv(Cpriv * kernel_x + C * (np.dot(kernel_x, kernel_z)) + kernel_z),(np.eye(size)*Cpriv + C * kernel_z)), T)
#         self.dual_alpha = np.dot(np.dot(np.linalg.inv(Cpriv * kernel_x + C * (np.dot(kernel_x, kernel_z)) + kernel_z+ np.eye(size)*C),(np.eye(size)*Cpriv + C * kernel_z)), T)

        self.support_vectors = X
        score1 = np.sum((T-np.dot(kernel_x,self.dual_alpha))**2,1)
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        score_test = self.threshold - np.sum((X-np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha))**2,1)
        
        return score_test
    
class AEKOC(object):
    '''
    This class implements AEKOC/AAKELM algorithm,
    
    Paper: Gautam, Chandan, Aruna Tiwari, and Qian Leng. "On the construction of extreme learning machine for online and offline one-class classification-An expanded toolbox."
    Neurocomputing 261 (2017): 126-143.
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.regularization = regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        size = X.shape[0]
        T = X
        self.dual_alpha = np.dot(np.linalg.inv(np.eye(size)*C + kernel_x), T)
        self.support_vectors = X
        score1 = np.sum((T-np.dot(kernel_x,self.dual_alpha))**2,1)
#         score1 = ((T-np.dot(kernel_x,self.dual_alpha))**2).sum(1)
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self
    
    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        score_test = self.threshold - np.sum((X-np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha))**2,1)
        
        return score_test
 
class ISVDD(object):
    '''
    This class implements SVDD+ algorithm,
    
    Paper: Zhang, Wenbo. "Support vector data description using privileged information."
    Electronics Letters 51.14 (2015): 1075-1076.
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 tol=0.001, max_iter=100, silent=True):
        # Setting initial parameters
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.privileged_regularization = privileged_regularization
        self.tol = tol
        self.silent = silent
        # Initializing with None some futer parameters
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None

    def _prepare_problem(self, X, Z):
        """
        Initializes optimization
        problem in form of several matrices
        for cvxopt framework
        """
        gamma = self.privileged_regularization
        C = 1.0 / len(X) / self.nu
        size = X.shape[0]
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        zeros_matrix = np.zeros_like(kernel_x)
        P = 2 * np.bmat([[kernel_x, zeros_matrix],
                         [zeros_matrix, 0.5*gamma*kernel_z]])
        P = matrix(P)
        q = matrix(list(np.diag(kernel_x)) + [0] * size)
        A = matrix([[1.]*size + [0.]*size, [1.] * size*2]).T
        b = matrix([1., 1.])
        G = np.bmat([[-np.eye(size), zeros_matrix],
                     [-np.eye(size), np.eye(size)],
                     [np.eye(size), -np.eye(size)]])
        G = matrix(G)
        G = sparse(G)
        h = matrix([0]*size*2 + [C]*size)
        optimization_problem = {'P': P, 'q': q, 'G': G,
                                'h': h, 'A': A, 'b': b}
        return optimization_problem

    # Helper function for prediction
    def _scalar_product_with_center(self, X):
        return np.dot(self.features_kernel(X, self.support_vectors),
                      self.dual_alpha)

    def _calculate_threshold(self):
        """
        Calculates critical value of the decision function
        """
        kernel_support = self.features_kernel(self.support_vectors)
        self.centre_norm = np.dot(self.dual_alpha,
                                  np.dot(kernel_support,
                                         self.dual_alpha))
        # Select the first support vector since distance
        # between it center equal to R
        single_support_vector = self.support_vectors[0, :]
        first_support_vector = single_support_vector.reshape(1, -1)
        support_vector_norm = self.features_kernel(first_support_vector)
        dot_product_with_centre = 2 * self._scalar_product_with_center(
                                          single_support_vector[np.newaxis, :])
        self.radius = (support_vector_norm +
                       self.centre_norm -
                       dot_product_with_centre)
        self.threshold = self.centre_norm - self.radius

    def fit(self, X, Z):
        '''
        Method takes matrix with feature values
        and information from privilaged feature
        space, solves optimization probem and
        calculate center and radius of sphere
        '''
        problem = self._prepare_problem(X, Z)
        options = {}
        options['show_progress'] = False #self.silent
        options['maxiters'] = self.max_iter
        options['abstol'] = self.tol
        problem['options'] = options
        solver = qp(**problem)
        if solver['status'] != 'optimal':
            raise ValueError("Failed Optimization")
        self.dual_solution = np.array(solver['x']).reshape(2*len(X),)
        self.support_indices = np.where(self.dual_solution[:len(X)] > 0)[0]
        self.support_vectors = X[self.support_indices, :]
        self.dual_alpha = self.dual_solution[self.support_indices]
        self._calculate_threshold()
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        test_norm = np.diag(self.features_kernel(X))
        scalar_product = self._scalar_product_with_center(X)
        return test_norm.ravel() + self.threshold - 2*scalar_product


class IOneClassSVM(object):
    '''
    This class implements OCSVM+ algorithm,
    
    Paper1: Zhu, Wenxin, and Ping Zhong. "A new one-class SVM based on hidden information."
    Knowledge-Based Systems 60 (2014): 35-43.
    
    Paper2: Burnaev, Evgeny, and Dmitry Smolyakov. "One-class SVM with privileged information and its application to malware detection."
    2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW). IEEE, 2016.
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 max_iter=100, tol=0.001):

        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.privileged_regularization = privileged_regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None

    def _prepare_problem(self, X, Z):
        """
        Initializes optimization
        problem in form of several matrices
        for cvxopt framework
        """
        gamma = self.privileged_regularization
        C = 1.0 / len(X) / self.nu
        size = X.shape[0]
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        q = np.zeros(len(X) * 2, dtype='float')
        q = matrix(q)
        zeros_matrix = np.zeros_like(kernel_x)
        P = 2 * np.bmat([[kernel_x, zeros_matrix],
                         [zeros_matrix, 0.5*gamma*kernel_z]])
        P = matrix(P)
        A = matrix([[1.]*size + [0.]*size, [1.] * size*2]).T
        b = matrix([1., 1.])
        G = np.bmat([[-np.eye(size), zeros_matrix],
                     [-np.eye(size), np.eye(size)],
                     [np.eye(size), -np.eye(size)]])
        G = matrix(G)
        G = sparse(G)
        h = matrix([0]*size*2 + [C]*size)
        optimization_problem = {'P': P, 'q': q, 'G': G,
                                'h': h, 'A': A, 'b': b}
        return optimization_problem

    def fit(self, X, Z):
        '''
        Method takes matrix with feature values
        and information from privilaged feature
        space, solves optimization probem and
        calculate center and radius of sphere
        '''
        problem = self._prepare_problem(X, Z)
        options = {}
        options['show_progress'] = False
        problem['options'] = options
        options['maxiters'] = self.max_iter
        options['abstol'] = self.tol
        solver = qp(**problem)
        if solver['status'] != 'optimal':
            raise ValueError("Failed Optimization")
        self.dual_solution = np.array(solver['x']).reshape(2*len(X),)
        self.support_indices = np.where(self.dual_solution[:len(X)] > 0)[0]
        self.support_vectors = X[self.support_indices, :]
        self.dual_alpha = self.dual_solution[self.support_indices]
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        return -np.dot(self.features_kernel(X, self.support_vectors),
                       self.dual_alpha)
 