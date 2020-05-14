'''
Methods
-------
(1) split_name
(2) vector_magnitude
(3) unicode_point
(4) unicode_array
(5) lcs
(6) max_lcs
(7) find_match
'''
import pandas as pd, numpy as np, time
import ipywidgets as widgets
from IPython.display import display

def split_name(s, t=['นาย','นางสาว','นาง','น.ส.','น.ส']):
    
    '''
    This function attempts to split name string into
    first and last name. Prior to splitting, if title 
    that matches the defined list is found, it will 
    be removed accordingly.
    
    Parameters
    ----------
    s : array_like of str or unicode
    
    t : list of str, optional, 
    default:['นาย','นางสาว','นาง','น.ส.','น.ส']
    \t List of titles to be removed from name.
    '''
    s = np.char.split(str(s)).tolist()
    k = np.array([s[0].find(t) for t in t])
    if (k==0).sum()>0:
        n = t[np.argmax(k==0)]
        if len(n)!=len(s[0]): 
            s[0] = s[0][s[0].find(n):len(n)]
        if (s[0]==n) & (len(s)>2): s = s[1:]
    if len(s)==1: s += ['']
    else: s = [s[0],' '.join(s[1:])]
    return s

def vector_magnitude(s, k=(3585,3663)):
    
    '''
    Find weighted character vectors and calculate
    corresponding magnitude. In addition, vector is
    determined by using ord() function, which returns
    an integer representing the Unicode code point of 
    that character. Every vector is assumed to have 
    its origin at starting unicode, k[0]-1.
    
    Parameters
    ----------
    s : array_like of str or unicode
    
    k : tuple of integers, optional, default:(3585,3663)
    \t Starting and ending unicode points, which starts
    \t from 'ก' (3685) until '๏' (3663).
    
    Returns
    -------
    magnitude of vectors : float
    
    Example
    -------
    >>> vector_magnitude('ทดลอง')
    12.432216214336043
    '''
    u0, cnt = np.unique(list(s), return_counts=True)
    u0 = np.array([ord(n) for n in u0])
    u = (u0>=k[0]) & (u0<=k[1])
    u0, cnt = u0[u], cnt[u]
    v = [cnt[n]*(u-k[0]-1)for n,u in enumerate(u0)]/sum(cnt)
    return np.sqrt(sum(v**2))

def unicode_point(s, k=(3585,3663)):
    
    '''
    Convert string into list of unicodes.
    
    Parameters
    ----------
    s : array_like of str or unicode
    
    k : tuple of integers, optional, default:(3585,3663)
    \t Starting and ending unicode points, which starts
    \t from 'ก' (3685) until '๏' (3663).
    '''
    u0 = np.arange(k[0],k[1])
    s = list(str(s).replace(' ',''))
    u = np.array([ord(n) for n in s])
    u = u[(u>=k[0]) & (u<=k[1])]
    u1, cnt = np.unique(u,return_counts=True)
    z = np.full(u0.shape,0.)
    z[np.isin(u0,u1)] = cnt
    return z.ravel()

def unicode_array(a, k=(3585,3663)):
    
    '''
    Convert all element in input array "a" into 
    integer matrix given range of unicode numbers
    
    Parameters
    ----------
    s : array of str or unicode
    \t List of texts
    
    k : tuple of integers, optional, default:(3585,3663)
    \t Starting and ending unicode points, which starts
    \t from 'ก' (3685) until '๏' (3663).
    '''
    if isinstance(a,str): a = [a]
    v = [unicode_point(s,k) for s in a]
    return np.vstack(v)

def lcs(s1, s2, remove=[' ']):
    
    '''
    * * Longest Common Subsequence * *
    It is the problem of finding the longest subsequence 
    common to all sequences in a set of sequences. In
    addition to subsequences, they are not required to 
    occupy consecutive positions within the original 
    sequences.
    
    reference
    ---------
    https://en.wikipedia.org/wiki/
    Longest_common_subsequence_problem
    
    Parameters
    ----------
    s1, s2 : array_like of str or unicode
    \t s1 and s2 can be in different lengths
    
    remove : list of char, optional, (default:[' '])
    \t List of characters that will be removed before
    \t finding subsequence.
    
    Returns
    -------
    Longest common subsequence : int
    
    Example
    -------
    >>> lcs('abc','cbnc')
    2
    '''
    def c_remove(s,r):
        s = np.array(list(s))
        return s[~np.isin(s,r)]
    
    s1 = c_remove(s1, remove)
    s2 = c_remove(s2, remove)

    len1, len2 = len(s1)+1, len(s2)+1 
    a = np.full((len1,len2),None)
    for i in range(len1): 
        for j in range(len2): 
            if (i==0)|(j==0): a[i,j] = 0
            elif s1[i-1] == s2[j-1]: a[i,j] = a[i-1,j-1] + 1
            else: a[i,j] = max(a[i-1,j],a[i,j-1]) 
    return a[-1,-1]

def max_lcs(s, a, remove=[' ']):
    
    '''
    Find maximum "Longest Common Subsequence" from
    input array "a"
    
    Parameters
    ----------
    s : array_like of str or unicode
    
    a : array of str
    
    remove : list of char, optional, (default:[' '])
    \t List of characters that will be removed before
    \t finding subsequence.

    Returns
    -------
    Dictionary of 
    - index : array of indices, whose "Longest Common 
              Subsequence" or "LCS" is the highest
    - text  : array of matched texts
    - score : maximum LCS score
    '''
    k = np.array([lcs(s,u,remove) for u in a])
    return dict(index=np.arange(len(a))[k==max(k)], 
                text=np.array(a)[k==max(k)], score=max(k))

def find_match(a, b, u=None, n=30, remove=[' '], sep=' , '):
    
    '''
    This function attempts to find the most similar
    text(s) from the input array "b". First of all, 
    unicode vectors are computed and by using
    euclidean distance, an array of proximities 
    between sample and input array is obtained.
    Subsequently, "n" closet elements are selected
    and undergo "LCS" process finding the pattern 
    score. The highest score will then be selected 
    as a match.
    
    Parameters
    ----------
    a : array of str, of shape (n_samples,)
    \t Array of strings or unicode to be tested
    \t against elements from input array "b".
    
    b : array of str, of shape (n_samples,)
    
    u : array of int, optional, (default:None)
    \t This array can be obtained from using 
    \t "unicode_array" method. If None, unicode
    \t array will be calculated from provided 
    \t input array "b".
    
    n : int, optional, (default:30)
    \t Number of texts that is similar to the 
    \t sample text. However, "n" could be larger
    \t than what defined if nth onward position 
    \t has the same proximity to the sample.
    
    remove : list of char, optional, (default:[' '])
    \t List of characters that will be removed
    \t before finding subsequence.
    
    sep : str, optional, (default=' , ')
    \t A separator that is used to join results 
    \t when "lcs" method returns more than one.
    
    Returns
    -------
    dictionary of pd.to_dict format
    {'text'   : [...], # matching text
     'score'  : [...], # 0 ≤ similarity score ≤ 1
     'index'  : [...], # index of similar texts
     'found'  : [...], # similar texts found
     'n_found': [...]} # number of results found
    '''
    t1, t2, t3 = progress_bar()
    if u is None:
        p = ['Calculating unicode martix',
             'Please wait this may take a few minutes']
        t1.value = ' , '.join(p)
        u = unicode_array(b)
        
    t1.value = 'Calculating . . .'
    start = time.time()
    k = np.arange(len(u))
    a0,a1,a2,a3,a4 = [],[],[],[],[]
    for c,s0 in enumerate(a):
        
        s = unicode_array(s0)
        d = np.sqrt((u-s)**2).sum(axis=1)
        # When there is no perfect match
        if min(d)>0:
            m = np.percentile(d,n/len(u)*100)
            i = k[d<=m].ravel()
        else: i = np.array([np.argmin(d)])
        x = max_lcs(s0, b[i], remove)
        
        a0.append(s0)
        n_max = max([len(t) for t in x['text']])
        a1.append(x['score']/float(n_max))
        a2.append(sep.join(i[x['index']].astype(str)))
        a3.append(sep.join(x['text']))
        a4.append(len(x['text']))
        
        pct = np.round((c+1)/len(a)*100,0)
        t2.value = '{:,.3g}%'.format(pct)
        avg_time = (time.time()-start)/(c+1)
        eta = hms(avg_time*(len(a)-c-1))
        t3.value = ' - ETA: {}  -  score: {:,.3g}'.format(eta, a1[c])
        
    t1.value = 'Complete . . .'
    return dict(text=a0, score=a1, index=a2, 
                found=a3, n_found=a4)

def progress_bar():
  
    t1 = widgets.HTMLMath(value='Calculating . . .')
    t2 = widgets.HTMLMath(value='{:,.3g}%'.format(0))
    s = ' - ETA: {}  -  {}: {:,.3g}'
    t3 = widgets.HTMLMath(value=s.format(np.nan,'metric',0))
    w = widgets.VBox([t1, widgets.HBox([t2,t3])])
    display(w); time.sleep(1)
    return t1, t2, t3

def hms(seconds):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return '[{:.0f}h : {:.0f}m : {:.0f}s]'.format(h, m, s)
