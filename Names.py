'''
Available methods are the followings:
[1] StringMatching

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 11-09-2023

'''
import numpy as np, re
from difflib import SequenceMatcher as sm
from itertools import product
from sklearn.utils import check_array

__all__ = ["StringMatching"]

class ValidateParams:
    
    '''Validate parameters'''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")

class StringMatching(ValidateParams):
    
    '''
    Determine the similarity between two names (case insensitive) by
    using `difflib.SequenceMatcher`.

    References
    ----------
    [1] https://docs.python.org/3/library/difflib.html
    [2] https://www.codexpedia.com/regex/regex-symbol-list-and-regex-examples/
    '''
    def __init__(self):
        self.criteria = {"english" : "[a-zA-Z][0-9]",
                         "thai"    : "[ก-๙]",
                         "symbol"  : "[!-/]|[:-@]|", 
                         "space"   : " * "}

    def __findall__(self, name:str, regex="symbol") -> bool:
        return re.findall(self.criteria[regex], name)

    def remove(self, name:str):
        
        '''Remove all symbols and whitespaces.'''
        # Remove all symbols
        adj_name = str(name)
        for symbol in np.unique(self.__findall__(adj_name)):
            adj_name = adj_name.replace(symbol,"")
        
        # Replace unnecssary spaces with single space
        while True:
            length = len(adj_name)
            spaces = self.__findall__(adj_name, "space")
            for symbol in np.unique(spaces):
                adj_name = adj_name.replace(symbol," ")
            if len(adj_name)==length: break
        return adj_name
    
    def __compare__(self, name, test_name, start=0):
        
        '''
        Alfter removing all symbols, it splits strings i.e. `name`, and 
        `test_name` into list of words by using space as a separator. 
        Then, it generates a cartesian product of words between `name` 
        and `test_name` and computes the maximum longest sequence ratio 
        with respect to words in `name`. Then, each ratio is weighted by 
        its length relative to the total string length and total score is 
        the sum of all weighted products.
        
        Parameters
        ----------
        name : str
            Input string. Any title or abbreviation should be removed 
            from `name` as to privide better comparison.

        test_name : str
            Test string against all words in `name`.

        start : int, default=0
            An index of word in `name` to start.
    
        Returns
        -------
        score : float
            The weighted ratios.
        '''
        # Initialize parameters
        ratios, weights = [], []
        
        # Split names.
        name1 = self.remove(test_name).lower().split(" ")
        name0 = self.remove(name).lower().split(" ")
        start = int(np.fmax(np.fmin(len(name0)-1, start), 0))

        for w0 in name0[start:]:
            weights+= [len(w0)]
            ratios += [max([sm(None, w0, w1).ratio() for w1 in name1])]
            
        # Calculate weighted score, and ratio.
        weights = np.r_[weights]/np.fmax(sum(weights),1)
        score = np.sum((ratios:=np.r_[ratios]) * weights)

        return score
        
    def compare(self, names, test_names, start=0):
        
        '''
        Compare two names (Case insensitive).
        
        Parameters
        ----------
        name : ndarray of shape (n_samples,)
            Input array of strings. Any title or abbreviation should be 
            removed from "name" as to privide better comparison.

        test_name : ndarray of shape (n_samples,)
            Test string against all words in `name`.
    
        start : int, default=0
            An index of word in `name` to start.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            An array of similarity scores. 
        
        Example
        -------
        >>> StringMatching().compare(['Sara' , 'Geoffrey'],
                                     ['Sarah', 'Jeffrey'])
        array([0.89, 0.80])
        '''
        
        # Input validation on an array
        kwds = dict(dtype=str, ensure_2d=False, copy=True)
        name0 = check_array(names, **kwds).flatten()
        name1 = check_array(test_names, **kwds).flatten()
        
        # Check length of arrays
        if len(name0)!=len(name1):
            raise ValueError(f'Inconsistent numbers of samples i.e. '
                             f'`names`={len(name0)}, and '
                             f'`test_names`={len(name1)}.')
        
        # Validate parameters
        start = self.Interval("start", start, int, 0, None, "left")
        
        # Calculate similariry scores
        scores = np.array([self.__compare__(n0, n1, start) 
                           for n0,n1 in zip(name0, name1)])
        
        return scores