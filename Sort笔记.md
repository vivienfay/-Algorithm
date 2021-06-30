### 总结

sort 方法
- bubble sort: O(n^2)
    * swap邻近两个元素
    
- selection sort: O(n^2)
    * 和最大元素进行调换
    
- quick sort：
    * average:O(nlogn) 最差 O(n^2)
    * 选第一个/最后一个/随机/中位数 做pivot
    * Using recursion
    * Choose a pivot , if the given number is smaller than pivot then put the number left to the pivot. Must to be inplace
    *     Firstly, put the pivot value into the last element
    * Subquestion: partition is used to put the number less than pivot before the pivot.
    * Inplace operation, thus space complexity is O(1)
    
- merge sort: devide and conqouer
    * Split the order in the middle till the level gets every element and adjust the order 
    * Merge the subarray according to their order
    * Subquestion: how to merge two arrays?
    * Using recursion(time complexity depends on node counts)
    
- insertion sort
- bucket sort

### 题型分类


- #### Bucket Sort
    - [How Many Numbers Are Smaller Than the Current Number](#1365)
    - [Maximum Gap](#164)

### 易错点


```python

```

### 必备模版技巧

### Quick Sort


```python
def quicksort(array, left, right):
    if left < right:
        pivot = partition(array,left,right)
        quicksort(array,left,pivot - 1)
        quicksort(array,pivot + 1, right)
    
def partition(array, left, right):
    pivot = array[left]
    i = left + 1
    for j in range(left+1, right+1):
        if array[j] < pivot:
            swap(array,i,j)
            i += 1
    swap(array,left,i-1)
    return i-1

def swap(nums, left, right):
    temp = nums[left]
    nums[left] = nums[right]
```


```python
arr = [70,80,20,30]
quicksort(arr,0,3)
arr
```




    [20, 30, 70, 80]



### Merge Sort


```python
def mergesort(array):
    mid = len(array) // 2
    if len(array)>1:
        L = array[:mid]
        R = array[mid:]
        
        mergesort(L)
        mergesort(R)
    
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = R[j]
                j += 1
            k += 1

        while i < len(L): 
            array[k] = L[i] 
            i+= 1
            k+= 1
          
        while j < len(R): 
            array[k] = R[j] 
            j+= 1
            k+= 1
```

### Insertion sort 


```python
def insertsort(nums):
    for i in range(1, len(nums)):
        cur = nums[i]
        j = i - 1
        while 0 <= j and nums[j] > cur:
            nums[i] = nums[j]
            j -=1
        nums[j+1] = cur
```

# 1365
[Leetcode](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/)
### How Many Numbers Are Smaller Than the Current Number


```python
class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        bucket = [0] * 101
        for i in nums: 
            bucket[i] +=  1
        counter = [0] * 101
        c = 0
        for ind, count in enumerate(bucket):
            if count != 0: 
                counter[ind] = c
                c += count
        return [counter[i] for i in nums]
```

# 692
[Leetcode](https://leetcode.com/problems/top-k-frequent-words/)
### Top K Frequent Words
可用heap做
或者quick selection做


```python
# 非最优解
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        map = {}
        for word in words:
            map[word] = map.get(word, 0) + 1
        
        res = sorted([(key, v) for key, v in map.items()], key = lambda x: (-x[1], x[0]))
        return [ i[0] for i in res[:k]]
```

# 1329
### Sort the Matrix Diagonally
[Leetcode](https://leetcode.com/problems/sort-the-matrix-diagonally/)


```python
class Solution(object):
    def diagonalSort(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        from collections import defaultdict
        m, n = len(mat), len(mat[0])
        d = defaultdict(list)
        for i in range(m):
            for j in range(n):
                d[i-j].append(mat[i][j])
        for k in d.keys():
            d[k].sort()
        for i in range(m):
            for j in range(n):
                mat[i][j] = d[i-j].pop(0)
        return mat
        
```

# 164

### Maximum Gap
[Leetcode](https://leetcode.com/problems/maximum-gap/)


```python
def maximumGap(self, nums):
        lo, hi, n = min(nums), max(nums), len(nums)
        if n <= 2 or hi == lo: return hi - lo
        B = defaultdict(list)
        for num in nums:
            ind = n-2 if num == hi else (num - lo)*(n-1)//(hi-lo)
            B[ind].append(num)
            
        cands = [[min(B[i]), max(B[i])] for i in range(n-1) if B[i]]
        return max(y[0]-x[1] for x,y in zip(cands, cands[1:]))
```

# 179
### Largest Number
[Leetcode](https://leetcode.com/problems/largest-number/)


```python

```
