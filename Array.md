# Array总结

## 总结

- 寻找连续的subarray的思路
  - 二分法
  - store prefix information in hash table, 例如prefix的index
  - sliding window
- 寻找subsequence思路
  - DP,生成两维数组
- Palindrome 相关题目思路
- substring 相关思路


1. 使用多个指针
    - 指针方向相反
    - 指针方向相同
    - 使用swap的方法

2. 转化成set

3. subarray
    - 二分法
    - store prefix information in hash table
    - sliding window
    - 
4. 使用hash map记住index

5. sliding window

6. #### interval题目:
    - 排序
    - greedy
    - 扫描线
    - DP

## 题型分类

### Two Pointer

### 先排序再做指针循环

- [3Sum](#15)
- [3Sum Closest](#16)
- [4Sum](#18)
- [3Sum Smaller](#259)

### 单指针

- [Maximum Subarray](#53)
- [Monotonic Array](#896)
- [Increasing Triplet Subsequence](#334)

### 双指针

- [Add Strings](#415)
- [Add Binary](#67)
- [Next Permutation](#34)
- [Best Time to Buy and Sell Stock II](#122)
- [Long Pressed Name](#925)
- [Intersection of Two array](#349)
- [Intersection of Three array]()

### swap
- [Rotate Image](#48)
- [Remove Duplicates from Sorted Array](#26)
- [Remove Duplicates from Sorted Array II](#80)
- [Move Zeros](#283)
- [Shortest Word Distance III](#245)
- [Minimum Swap](#670)


1. 快慢指针
19， 141， 142， 234， 457， 287


4. 反向指针

    - [Two Sum Less Than K](#1099)
    - [Valid Palindrom](#125)
    - [Valid palindrome II](#680)
    - [Squares of a Sorted Array](#977)
    - [Reverse String](#344)
    - [Reverse Vowels of a String](#345)
    - [Sort Colors](#75)
    - [Statistics from a large Sample](#1093)
    - [Boats to Save People](#881)


5. 应用于有序数列
977, 360, 532, 881, 167, 15, 16, 259, 923, 18

### Presum/Prefix Product

- [Maximum Subarray](#53)
- [Subarray Sum Equals K](#560)
- [Subarray Sums Divisible by K](#974)
- [Product of the Last K Numbers](#1352)
- [Range Sum Query - Immutable](#303)
- [Maximum Size Subarray Sum Equals k](#maximum-size-subarray-sum-equals-k)
- [Range Addition](#range-addition)
    
### two direction

- [Product of Array Except Self](#238)
- [Trapping Rain Water](#42)
    
### swap
- [Rotate Image](#48)
- [Remove Duplicates from Sorted Array](#26)
- [Remove Duplicates from Sorted Array II](#80)
- [Move Zeros](#283)
- [Shortest Word Distance III](#245)
- [Minimum Swap](#670)
    
### 字符
- [Group shifted Strings](#249)

### 记录特别的变量

- [Best Time to Buy and Sell Stock](#121)
- [Shortest Unsorted Continuous Subarray](#581)
    
### 三指针
    - [Intersection of Three Sorted Arrays](#1214)
    
### 用-1做标记
    - [Find All Duplicates in an Array](#442)
    - [Find All Numbers Disappeared in an Array](#448)
    
### hashmap

- [Two Sum]()
- [Number of Good Pairs](#1512)
- [Partition Labels](#763)
- [Longest Consecutive Sequence](#128)
- [Group Anagram](#)

### Greedy

- [Queue Reconstruction by Height](#406)

### Sweep Line

- [The Skyline Problem](#the-skyline-problem)

### 解体思路总结

### 易错点

### 必备模版技巧

----
### [Two Sum](https://leetcode.com/problems/two-sum/)

- 由于需要输出index，使用hash dict记住index
- O（n）

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        map = {}
        for i, v in enumerate(nums):
            if v in map.keys(): return [i, map[v]]
            map[target-v] = i
        return -1
```

### [Two Sum Less Than K](https://leetcode.com/problems/two-sum-less-than-k/)

- 使用两个指针 反向缩小区间
- O(nlogn)

```python
class Solution(object):
    def twoSumLessThanK(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        A.sort()
        max_sum = -1
        l = 0
        r = len(A) - 1
        while l < r:
            if A[l] + A[r] < K:
                max_sum = max(max_sum, A[l] + A[r])
                l = l + 1 
            else:
                r = r - 1
        return max_sum
```

### [Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/)

- O(n)
- 使用 <= 是因为要和 > 做比较

```python
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # first represents the smallest value, second: the second smallest value
        first = second = float('inf')
        for i in nums:
            if i <= first:
                first = i
            elif i <= second:
                second = i
            else: return True
        return False
```

### [Three Sum](https://leetcode.com/problems/3sum/)

- O(n**2)
- 三个指针轮番走
- 注意unique，可以while循环remove 重复值

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            return []
        
        result = []
        nums.sort()
        
        for p in range(len(nums) - 2):
            l, r = p + 1, len(nums) - 1
            if nums[p] == nums[p-1] and p >= 1: continue
            while l < r:
                if nums[l] + nums[r] == -nums[p]:
                    result.append([nums[p],nums[l],nums[r]])
                    while nums[l] == nums[l + 1] and l + 1 < r:
                        l += 1
                    while nums[r] == nums[r - 1] and r - 1 > l:
                        r -= 1
                
                    l += 1
                    r -= 1
                
                elif nums[l] + nums[r] > -nums[p]:
                    r -= 1
                else:
                    l += 1
                
                
        return result
```

### [3Sum Closest](https://leetcode.com/problems/3sum-closest/)

```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        closest = float('inf')
        nums = sorted(nums)
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                new_sum = nums[i] + nums[l] + nums[r]
                if abs(new_sum - target) < abs(closest - target):
                    closest = new_sum
                if new_sum > target:
                    r -= 1
                elif new_sum < target:
                    l += 1
                else:
                    return new_sum
        return closest   
```

### [4 Sum](https://leetcode.com/problems/4sum/submissions/)

- 注意duplicate的时候 用while移动指针

```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(nums) < 4: return []
        nums = sorted(nums)
        res = []
        for i in range(0, len(nums) - 3):
            if nums[i] == nums[i - 1] and i > 0: continue
            for j in range(i + 1, len(nums) - 2):
                if nums[j] == nums[j - 1] and j > i + 1: continue
                l, r = j + 1, len(nums) - 1
                while l < r:
                    new_sum = nums[i] + nums[j] + nums[l] + nums[r]
                    if new_sum == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while nums[l] == nums[l + 1] and l + 1 < r: l += 1
                        while nums[r] == nums[r - 1] and l < r - 1: r -= 1
                        l += 1; r -= 1
                    elif new_sum > target: r -= 1
                    elif new_sum < target: l += 1
        return res
```

### [Move Zeros](https://leetcode.com/problems/move-zeroes/)

- swap的方法，把不要的东西挪到后面去
- 两个指针，一个用来标记在它前面都要的东西，一个用来标记需要比较的对象。需要的对象收到第一指针里面去。

```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1
```

### [3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)

```python
class Solution(object):
    def threeSumSmaller(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums = sorted(nums)
        res = 0
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                if nums[i] + nums[l] + nums[r] < target:
                    res += r - l
                    l += 1
                else:
                    r -= 1
        return res
        
```

### [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

- 两个指针，不断swap


```python
class Solution(object):
    def swap(self, nums, i, j):
        temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
        
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        i, j = 0, 0
        
        while j < len(nums):
            if nums[j] > nums[i]:
                self.swap(nums,i+1,j)
                i += 1
                j += 1
            else:
                j += 1
                
                
        return i + 1
```

### [Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tail = 0
        for num in nums:
            if tail < 2 or num != nums[tail - 1] or num != nums[tail - 2]:
                nums[tail] = num
                tail += 1
        return tail
```

### [Missing Number](https://leetcode.com/problems/missing-number/)


```python
class Solution(object):
    def missingNumber(self, nums):
        return len(nums) * (len(nums) + 1) / 2 - sum(nums)
```

### Maximum Subarray

### Subarray Sum

### merge two sorted array

```python
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        curr = dummy
        while l1 and l2:
            if l1.val > l2.val:
                curr.next = l2
                l2 = l2.next    
            else:
                curr.next = l1
                l1 = l1.next
            curr = curr.next
            
        curr.next = l1 if l1 else l2
        
        return dummy.next
```

### [intersection of two arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        nums1.sort()
        nums2.sort()
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                res.append(nums1[i])
                while i + 1 < len(nums1) and nums1[i] == nums1[i + 1]: i += 1
                while j + 1 < len(nums2) and nums2[j] == nums2[j + 1]: j += 1
                i += 1; j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1
        return res
```

```python
class Solution:
    def intersection(self, nums1, nums2):
        nums1 = set(nums1)
        res = set()
        for i in nums2:
            if i in nums1:
                res.add(i)
                
        return res
```

### median of two sorted array

### sort list

### [K-diff Pairs in an Array](https://leetcode.com/problems/k-diff-pairs-in-an-array/)

- 考虑edge case, 如果k=0要如何处理

```python
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        c = collections.Counter(nums)
        res = 0
        for i in c:
            if k > 0 and i + k in c or k == 0 and c[i] > 1:
                res += 1
        return res
```

### partition array

### Sort letters by oven and even

### Sort letters by case

### [Sort Colors](https://leetcode.com/problems/sort-colors/)

```python
class Solution:
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        l, r, i = 0, len(nums) - 1, 0
        while i <= r:
            print(i)
            print(nums)
            if nums[i] == 0:
                nums[i], nums[l] = nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] == 1:
                i += 1
            else:
                nums[r], nums[i] = nums[i], nums[r]
                r -= 1
        return nums
```

### Rainbow Sort

### pancake sort

### [Container with most Water](https://leetcode.com/problems/container-with-most-water/)


```python
class Solution:
    def maxArea(self, height):
        i, j = 0, len(height) - 1
        max_area = -float('inf')
        while i < j:
            max_area = max(max_area, min(height[i], height[j]) * (j - i))
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1
        return max_area
```

### [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

- 两个指针是从后面开始遍历的
- 考虑如果一个走完了，另一个怎么办

```python
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        ind = m + n - 1
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[ind] = nums1[m - 1]
                ind -= 1
                m -= 1
            else:
                nums1[ind] = nums2[n - 1]
                ind -= 1
                n -= 1
        while n > 0:
            nums1[ind] = nums2[n - 1]
            ind -= 1
            n -= 1
        while m > 0:
            nums1[ind] = nums1[m - 1]
            ind -= 1
            m -= 1
        return nums1
```

### [Meeting Room II](https://leetcode.com/problems/meeting-rooms-ii/)

- 贪心算法
- 也可用扫描线

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        start, end = [], []
        for i, j in intervals:
            start.append(i)
            end.append(j)
        start = sorted(start)
        end = sorted(end)
        j = 0
        count = 0
        for i in range(len(intervals)):
            print(count)
            count += 1
            if start[i] >= end[j]:
                j += 1
                count -= 1
        return count
                
```

### [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

- 以index为中心往左右散开test是不是palindromic

```python
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        for i in range(len(s)):
            res += self.Palindrom(s, i, i)
            if i + 1 < len(s) : res += self.Palindrom(s, i, i + 1)
        return res
   
    
    def Palindrom(self, s, start, end):
        count = 0
        while 0 <= start and end < len(s):
            if s[start] == s[end]: 
                count += 1
                start -= 1
                end += 1
            else: break
        return count     
```

### [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ''
        for i in range(len(s)):
            res = self.helper(s, res, i, i)
            res = self.helper(s, res, i, i + 1)
        return res
    
    def helper(self, s, res, start, end):
        while 0 <= start and end < len(s):
            if s[start] == s[end]:
                if len(s[start: end + 1]) > len(res):
                    res = s[start:end + 1]
                start -= 1
                end += 1
            else: break
        return res
```

### [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

- presum

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curSum = maxSum = nums[0]
        for num in nums[1:]:
            curSum = max(curSum + num, num)
            maxSum = max(maxSum, curSum)
        return maxSum
```

### [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

- presum = hashmap

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = 0
        sums = 0
        d = dict()
        d[0] = 1
        
        for i in range(len(nums)):
            sums += nums[i]
            count += d.get(sums-k,0)
            d[sums] = d.get(sums,0) + 1
        
        return(count)
```

### [Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)

- presum + hashmap

```python
class Solution(object):
    def subarraysDivByK(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        d = {0:1}
        sum, count = 0, 0
        for i in range(len(A)):
            sum = (sum + A[i]) % K
            count += d.get(sum, 0)
            d[sum] = d.get(sum, 0) + 1
        return count
```

### [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        left = 1
        for i in range(len(nums)):
            res.append(left)
            left *= nums[i]
        right = 1
        for j in range(len(nums) - 1, -1, -1):
            res[j] = res[j] * right
            right *= nums[j]
        return res
```

### [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

以最高点为中点，两个方向向中间移动

```python
# O(n)
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height: return 0
        left, right = [], []
        lmax, rmax = height[0], height[-1]
        for i in range(len(height)):
            lmax = max(height[i], lmax)
            left.append(lmax)
        for i in range(len(height) - 1, -1, -1):
            rmax = max(height[i], rmax)
            right = [rmax] + right
        
        res = 0
        for i, j, k in zip(left, right, height):
            res += min(i,j) - k
        return res   
```

```python
# O(1)
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height: return 0
        highest_index = height.index(max(height))
        lmax = height[0]
        rmax = height[-1]
        res = 0
        for i in range(0, highest_index):
            lmax = max(lmax, height[i])
            res += min(lmax, height[highest_index]) - height[i]
        for i in range(len(height) - 1, highest_index, -1):
            rmax = max(rmax, height[i])
            res += min(rmax, height[highest_index]) - height[i]

        return res   
```

### [Rotate Image](https://leetcode.com/problems/rotate-image/)

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n/2):
            for j in range(n):
                matrix[i][j], matrix[n-i-1][j] = matrix[n-i-1][j], matrix[i][j]
        for i in range(n):
            for j in range(n):
                if j < i: matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
        return matrix
```

### [Add Strings](https://leetcode.com/problems/add-strings/)

```python
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        res = ''
        sum, reminder = 0, 0
        i, j = len(num1) - 1, len(num2) - 1
        while 0 <= i or 0 <= j or reminder != 0:
            n1 = n2 = 0
            if i >= 0: n1 = int(num1[i])
            if j >= 0: n2 = int(num2[j])
            sum = n1 + n2 + reminder
            reminder = sum / 10
            res = str(sum % 10) + res
            i -= 1
            j -= 1
        return res
```

### [Add Binary](https://leetcode.com/problems/add-binary/)


```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        i, j = len(a) - 1, len(b) - 1
        res = ''
        reminder = 0
        while i >= 0 and j >= 0:
            sum = int(a[i]) + int(b[j]) + reminder
            val = sum % 2
            reminder = sum // 2
            res = str(val) + res
            i -= 1
            j -= 1
        if j >= 0:
            a = b
            i = j
        while i >= 0:
            sum = reminder + int(a[i])
            val = sum % 2
            reminder = sum // 2
            res = str(val) + res
            i -= 1
        if reminder != 0:
            res = str(reminder) + res
        return res
```

### [Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/)

```python
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        d = {v: ind for ind, v in enumerate(order)}
        for a, b in zip(words, words[1:]):
            ind = 0
            while ind < len(a) and ind < len(b):
                if d[a[ind]] > d[b[ind]]: return False
                elif d[a[ind]] < d[b[ind]]: break
                ind += 1
            if ind < len(a) and ind >= len(b): return False
        return True
                
```

### [Next Permutation](https://leetcode.com/problems/next-permutation/)

```python
class Solution(object):
    def nextPermutation(self, nums):
        i = j = len(nums)-1
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        if i == 0:   
            nums.reverse()
            return 
        k = i - 1  
        while nums[j] <= nums[k]:
            j -= 1
        nums[k], nums[j] = nums[j], nums[k]  
        l, r = k+1, len(nums)-1 
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l +=1 ; r -= 1
        return nums
```

### [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res = []
        while matrix:
            res.extend(matrix.pop(0))
            if matrix:
                for i in range(len(matrix)):
                    if matrix[i]:
                        res.append(matrix[i].pop())
                        
            if matrix:
                res.extend(matrix.pop()[::-1])
            if matrix:
                for i in range(len(matrix)-1, -1, -1):
                    if matrix[i]:
                        res.append(matrix[i].pop(0))
                        
        return res
```

```python
# two pointer
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        row_s, row_e, col_s, col_e = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        res = []
        while len(res) != len(matrix) * len(matrix[0]):
            for i in range(col_s, col_e + 1):
                res.append(matrix[row_s][i])
            row_s += 1
            for i in range(row_s, row_e + 1):
                res.append(matrix[i][col_e])
            col_e -= 1 
            if len(res) == len(matrix) * len(matrix[0]): break
            for i in range(col_e, col_s - 1, -1):
                res.append(matrix[row_e][i])
            row_e -= 1    
            for i in range(row_e, row_s -1, -1):
                res.append(matrix[i][col_s])
            col_s += 1  
        return res
            
```

### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

- 记录要存储的变量， 能够使得他成为o（N）解法

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        min_price = float('inf')
        for price in prices:
            min_price = min(min_price, price)
            res = max(price - min_price, res)
        return res
```

### [String Compression](https://leetcode.com/problems/string-compression/)

```python
class Solution(object):
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        st = i = 0
        while i < len(chars):
            while i < len(chars) and chars[i] == chars[st]:
                i += 1
            if i - st == 1:
                st = i
            else:
                print(chars)
                chars[st + 1 : i] = str(i - st)
                print(chars)
                st = st + 2
                i = st
```

### [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
-双指针


```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        i = j = 0
        res = 0
        while i < len(prices) and j < len(prices):
            if j == 0 or prices[j - 1] <= prices[j]:
                j += 1
            else:
                res += prices[j - 1] - prices[i]
                i = j
                j += 1
        res += prices[j - 1] - prices[i] if prices[j - 1] > prices[i] else 0
        return res
```

### [Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)


```python
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        n1, n2 = sorted(nums1), sorted(nums2)
        i = j = 0
        res = []
        while i < len(n1) and j < len(n2):
            if n1[i] < n2[j]:
                i += 1
            elif n1[i] > n2[j]:
                j += 1
            else:
                res.append(n1[i])
                i += 1
                j += 1
        return res
```

### [Permutation in String](https://leetcode.com/problems/permutation-in-string/)

```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        from collections import Counter
        d1 = Counter(s1)
        d2 = {}
        i = j = 0
        while j < len(s2): 
            num = s2[j]
            d2[num] = d2.get(num, 0) + 1
            if j - i + 1 == len(s1):
                if d2 == d1: return True
                d2[s2[i]] -= 1
                if d2[s2[i]] == 0: del d2[s2[i]]
                i += 1
            j += 1
        return False
```

### [Minimum Swaps to Group All 1's Together](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/)


```python
class Solution(object):
    def minSwaps(self, data):
        """
        :type data: List[int]
        :rtype: int
        """
        cnt1 = sum(data)
        i = j = 0
        total1 = 0
        res = len(data)
        while j < len(data):
            if data[j] == 1:
                total1 += 1
                cnt1 -= 1
            j += 1
            while i < j and (j-i-total1) > cnt1:
                if data[i] == 1: 
                    total1 -= 1
                    cnt1 += 1
                i += 1
            if j - i - total1 == cnt1: res = min(res, j - i - total1)
        return res     
```

### [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)

```python
class Solution(object):
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        ma
        count = 0
        sum = 0
        res = 0
        while j < len(tree):
            map[tree[j]] = map.get(tree[j], 0) + 1
            if map[tree[j]] == 1: count += 1
            sum += tree[j]
            j += 1
            while count > 2:
                map[tree[i]] -= 1
                if map[tree[i]] == 0: count -= 1
                i += 1
            res = max(res, j - i)
        return res
```

### [Swap For Longest Repeated Character Substring](https://leetcode.com/problems/swap-for-longest-repeated-character-substring/)

- 用一个inner outer分别记录区间内和区间外的的情况
- max——ch记录众数的情况
- 要考虑两种情况：需要替换或者不需要替换

```python
class Solution(object):
    def maxRepOpt1(self, text):
        """
        :type text: str
        :rtype: int
        """
#         a a b a bbbb
        from collections import Counter
        outer = Counter(text)
        inner = {}
        res = 0
        max_ch, max_num = '', 0
        i = j = 0
        while j < len(text):
            inner[text[j]] = inner.get(text[j], 0 ) + 1
            if inner[text[j]] > max_num: max_ch, max_num = text[j], inner[text[j]]
            j += 1
            while j - i - max_num > 1:
                inner[text[i]] -= 1
                i += 1
            res = max(res, min(j -i, outer[max_ch]))
        return res
```

### [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

```python
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        d = {}
        count = 0
        res = 0
        major_len = 0
        i = j = 0
        while j < len(s):
            d[s[j]] = d.get(s[j], 0) + 1
            major_len = max(major_len, d[s[j]])
            j += 1
            while j-i-major_len > k:
                d[s[i]] -= 1
                i += 1
            res = max(res, j-i)
        return res
```

### [Find K-Length Substrings With No Repeated Characters](https://leetcode.com/problems/find-k-length-substrings-with-no-repeated-characters/)

```python
class Solution(object):
    def numKLenSubstrNoRepeats(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: int
        """
        d = {} 
        i = j = 0
        count = 0 
        while j < len(S):
            d[S[j]] = d.get(S[j], 0) + 1
            j += 1
            if len(d) < j - i or j - i > K:
                d[S[i]] -= 1
                if d[S[i]] == 0: del d[S[i]]
                i += 1
            if len(d) == K == j - i: count += 1
            # print(d, S[i], S[j], count)
            # print(i, j)
        return count
```

### [Long Pressed Name](https://leetcode.com/problems/long-pressed-name/)

```python
class Solution(object):
    def isLongPressedName(self, name, typed):
        """
        :type name: str
        :type typed: str
        :rtype: bool
        """
        i = 0
        for j in range(len(typed)):
            if i < len(name) and name[i] == typed[j]:
                i += 1
            elif j == 0 or typed[j] != typed[j - 1]:
                return False
        return i == len(name)
```

### [Intersection of Three Sorted Arrays](https://leetcode.com/problems/intersection-of-three-sorted-arrays/)

```python
class Solution(object):
    def arraysIntersection(self, arr1, arr2, arr3):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type arr3: List[int]
        :rtype: List[int]
        """
        i = j = k = 0
        res = []
        while i < len(arr1) and j <len(arr2) and k < len(arr3):
            if arr1[i] == arr2[j] == arr3[k]:
                res.append(arr1[i])
                i += 1
                j += 1
                k += 1
            elif min(arr1[i], arr2[j], arr3[k]) == arr1[i]: i += 1
            elif min(arr1[i], arr2[j], arr3[k]) == arr2[j]: j += 1
            elif min(arr1[i], arr2[j], arr3[k]) == arr3[k]: k += 1
        return res
        
```

Maximum Product Subarray


```python
class Solution(object):
    def maxProduct(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(A + B)        
        
```

### [Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

用-1给index做标记

```python
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for x in nums:
            if nums[abs(x) - 1] < 0:
                res.append(abs(x))
            else:
                nums[abs(x) - 1] *= -1
        return res
```

### [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

```python
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in xrange(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]
       
```

### [Jump Game](https://leetcode.com/problems/jump-game/)

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i+n)
        return True        
```

### [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower()!= s[j].lower(): 
                return False
            i += 1
            j -=1
        return True
```

### [Monotonic Array](https://leetcode.com/problems/monotonic-array/)

```python
class Solution(object):
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        last = A[0]
        increasing = decreasing = True
        for i in A[1:]:
            if last > i: increasing = False
            if last < i: decreasing = False
            if not (increasing or decreasing): return False
            last = i
        return True
        
```

### [Number of Good Pairs](https://leetcode.com/problems/number-of-good-pairs/)

```python
class Solution(object):
    def numIdenticalPairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = {}
        for i in nums:
            d[i] = d.get(i, 0) + 1
        res = 0
        for k, v in d.items():
            res += (v-1) * v / 2
        return res
```

### [Partition Labels](https://leetcode.com/problems/partition-labels/)

```python
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        d = {}
        for i in S:
            d[i] = d.get(i, 0) + 1
        i = j = 0
        s = set()
        res = []
        t = set()
        while j < len(S):
            d[S[j]] -= 1
            if d[S[j]] == 0: t.add(S[j])
            s.add(S[j])
            if len(s) == len(t): 
                res.append(j-i + 1)
                s, t = set(), set()
                i = j + 1
            j += 1
        return res
            
```

### [Robot Bounded In Circle](https://leetcode.com/problems/robot-bounded-in-circle/)

```python
class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        x, y = 0, 0
        dx, dy = 0, 1
        for i in instructions:
            if i == 'G': x, y = x + dx, y + dy
            elif i == 'L': dx, dy = -dy, dx
            else: dx, dy = dy, -dx
        return (x==0 and y==0) or (dx, dy) != (0,1)
```

### [Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)

```python
class Solution(object):
    def minOperations(self, nums, x):
        """
        :type nums: List[int]
        :type x: int
        :rtype: int
        """
        i, j = 0, 0
        total = sum(nums)
        count = 0
        res = -1
        while j < len(nums):
            count += nums[j]
            j += 1
            while i < j and total - count < x:
                count -= nums[i]
                i += 1
            if total - count == x:
                res = max(res, j - i)
            # print(i, j, count, x, res)
        return len(nums) - res if res != -1 else -1
            
```

### [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

```python
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = {}
        for i in nums:
            if i in d: continue
            up = down = 0
            if i + 1 in d: up += d[i+1]
            if i - 1 in d: down += d[i-1]
            d[i] = up + down + 1
            if i - 1 in d:d[i-d[i-1]] = d[i]
            if i + 1 in d:d[i+d[i+1]] = d[i]
        return max(d.values()) if d else 0
```

### [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)

```python
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        n = len(nums)
        j = 0
        while j < n and nums[j] < 0:
            j += 1
        i = j - 1
        while 0 <= i and j < n:
            if nums[i] ** 2 < nums[j] ** 2:
                res.append(nums[i] ** 2)
                i -= 1
            else:
                res.append(nums[j] ** 2)
                j += 1
        while i >= 0:
            res.append(nums[i] ** 2)
            i -= 1
        while j < n:
            res.append(nums[j] ** 2)
            j += 1
        return res
                
```

### [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

```python
class Solution(object):
    def validPalindrome(self, s):
        i = 0
        while i < len(s) / 2 and s[i] == s[-(i + 1)]: i += 1
        s = s[i:len(s) - i]
        return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]
```

### [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

```python
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        import heapq
        from collections import Counter
        d = Counter(nums)
        h = [(-v, key) for key, v in d.items()]
        heapq.heapify(h)
        res = []
        for i in range(k):
            key, val = heapq.heappop(h)
            res.append(val)
        return res
```

### [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        d = defaultdict(list)
        for str in strs:
            s = tuple(sorted([i for i in str]))
            d[s].append(str)
        return [v for k, v in d.items()]
            
```

### [Group Shifted mStrings](https://leetcode.com/problems/group-shifted-strings/)

```python
class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        def shift(str):
            res = [ord(i) - ord('a') for i in str]
            diff = 26 - res[0]
            shift = [i+diff if 0 < i + diff <= 26 else i + diff - 26 for i in res]
            return ''.join([chr(i + ord('a')) for i in shift])
            
        
        from collections import defaultdict
        d = defaultdict(list)
        for s in strings:
            shifted = shift(s)
            d[shifted].append(s)
        return [v for k, v in d.items()]  
```

### [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people = sorted(people, key = lambda x: (-x[0], x[1]))
        res = []
        for p in people:
            res.insert(p[1], p)
        return res
```

### [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def partition(nums, left, right):
            p = nums[left]
            i = left + 1
            for j in range(i, right + 1):
                if nums[j] < p:
                    nums[j], nums[i] = nums[i], nums[j]
                    i += 1
            nums[left], nums[i - 1] = nums[i-1],nums[left]
            return i - 1
            
        def findk(nums, left, right, k):
            if left == right: return nums[left]
            if left < right:
                pivot = partition(nums, left, right)
                if pivot < k:
                    return findk(nums,pivot + 1, right, k)
                elif pivot > k:
                    return findk(nums, left, pivot - 1, k)
                else:
                    print(nums, pivot, k)
                    return nums[pivot] 
        
        if len(nums) < k: return 
        return findk(nums, 0, len(nums)-1, len(nums)-k)
```

### [Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

```python
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        mmax, mmin = nums[0], nums[-1]
        start = end = float('inf')
        i, j = 0, len(nums) - 1
        while i < len(nums):
            mmax = max(nums[i], mmax)
            if mmax > nums[i]: end = i
            i += 1
        while 0 <= j:
            mmin = min(nums[j], mmin)
            if mmin < nums[j]: start = j
            j -= 1
        return end - start + 1 if start != float('inf') else 0
```

### [Boats to Save People](https://leetcode.com/problems/boats-to-save-people/)

```python
class Solution(object):
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort(reverse=True)
        i, j = 0, len(people) - 1
        while i <= j:
            if people[i] + people[j] <= limit: j -= 1
            i += 1
        return i
```

### [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)

```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        m, n = len(board), len(board[0])
        for j in range(n):
            s = [board[i][j] for i in range(m) if board[i][j] != '.']
            if len(s) != len(set(s)): return False
        for i in range(m):
            s = [board[i][j] for j in range(n) if board[i][j] != '.']
            if len(s) != len(set(s)): return False
        for i in range(m):
            for j in range(n):
                if i % 3 == 0 and j % 3 == 0:
                    s = [board[x][y] for x in range(i, i+3) for y in range(j, j+3) if board[x][y] != '.']
                    if len(s) != len(set(s)): return False
        return True
```

### [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        n = len(nums)
        if len == 0: return []
        if k == 0: return nums
        deque = collections.deque()
        res =[]
        for i in range(0, n):
            if i < k-1:
                while deque and deque[-1] < nums[i]: deque.pop()
                deque.append(nums[i])
            else:
                while deque and deque[-1] < nums[i]:deque.pop()
                deque.append(nums[i])
                res.append(deque[0])
                if deque[0] == nums[i-k+1]:deque.popleft()
        return res
```

### [Reorganize String](https://leetcode.com/problems/reorganize-string/)

- 使用heapq能够帮助形成greedy的机制

```python
class Solution(object):
    def reorganizeString(self, S):
        """
        :type S: str
        :rtype: str
        """
        from collections import Counter
        import heapq
        d = Counter(S)
        h = [(-v, k)for k, v in d.items()] 
        res = ''
        heapq.heapify(h)
        while len(h) > 1:
            v1, k1 = heapq.heappop(h)
            v2, k2 = heapq.heappop(h)
            res += k1 + k2
            if v1 + 1!= 0: heapq.heappush(h, (v1+1, k1))
            if v2 + 1!= 0: heapq.heappush(h, (v2+1, k2))  
        if len(h) == 0: return res
        else:
            v1, k1 = heapq.heappop(h)
            if v1 == -1: return res + k1
            else: return ''
```

### [Task Scheduler](https://leetcode.com/problems/task-scheduler/)

```python
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter
        import heapq
        d = Counter(tasks)
        h = [(-v, k) for k, v in d.items()]
        heapq.heapify(h)
        count = 0
        print(h)
        while len(h) > 0:
            num_ls = []
            count_ls = []
            for _ in range(n+1):
                if len(h) > 0:
                    v, k = heapq.heappop(h)
                    num_ls.append(k)
                    count_ls.append(v + 1)
                else: count += 1
            if sum(count_ls) == 0: count -= n +1 - len(num_ls)
            for i, j in zip(num_ls, count_ls):
                if j!=0: heapq.heappush(h, (j, i))
        return len(tasks) + count

```

### [Maximum Swap](https://leetcode.com/problems/maximum-swap/)

```python
A = map(int, str(num))
last = {x: i for i, x in enumerate(A)}
for i, x in enumerate(A):
    for d in xrange(9, x, -1):
        if last.get(d, None) > i:
            A[i], A[last[d]] = A[last[d]], A[i]
            return int("".join(map(str, A)))
return num
```

### [Minimum Cost to Connect Sticks](https://leetcode.com/problems/minimum-cost-to-connect-sticks/)

```python
class Solution(object):
    def connectSticks(self, sticks):
        """
        :type sticks: List[int]
        :rtype: int
        """
        import heapq
        heapq.heapify(sticks)
        res = 0
        while len(sticks) > 1:
            a = heapq.heappop(sticks)
            b = heapq.heappop(sticks)
            heapq.heappush(sticks, a+b)
            res += a+b
        return res
        
```

### [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

```python
class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        for i in range(len(points)):
            x, y = points[i]
            points[i] = [x**2 +y** 2, x,y,]
        heapq.heapify(points)
        return [heapq.heappop(points)[1:] for _ in range(K)]
            
```

### [Game of Life](https://leetcode.com/problems/game-of-life/)

```python
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                count = 0
                for dx,dy in [(1, 0),(-1, 0),(0, 1),(0, -1),(1, 1),(1, -1),(-1, 1),(-1, -1)]:
                    if 0 <= i + dx < m and 0 <= j + dy < n:
                        count += board[i + dx][j + dy] % 10
                board[i][j] = count * 10 + board[i][j]
        for i in range(m):
            for j in range(n):
                if board[i][j] % 10 == 0:
                    if board[i][j] == 30: board[i][j] = 1
                    else: board[i][j] = 0
                else:
                    if board[i][j] == 21 or board[i][j] == 31: board[i][j] = 1
                    else: board[i][j] = 0
        return board
```

### [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

```python
from heapq import *
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = []
        self.large = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else: return self.large[0]

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

### [Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/)

重点： 对于0的情况应该重新清理


```python
class ProductOfNumbers(object):

    def __init__(self):
        self.q = [1]

    def add(self, num):
        """
        :type num: int
        :rtype: None
        """
        if num == 0: self.q = [1]
        else: self.q.append(num * self.q[-1])

    def getProduct(self, k):
        """
        :type k: int
        :rtype: int
        """
        if k >= len(self.q): return 0
        return self.q[-1] / self.q[-k-1]
```

### [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)

```python
class NumArray:

    def __init__(self, nums: List[int]):
        s = 0
        self.sum_num = []
        self.nums = nums
        for i in nums:
            self.sum_num.append(s)
            s += i
        self.sum_num.append(s)

    def sumRange(self, left: int, right: int) -> int:
        return self.sum_num[right+1] - self.sum_num[left]
```

### [Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)

```python
class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Node':
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        while node.parent:
            if node.parent.left == node:
                return node.parent
            node = node.parent
        return None
```

### [Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

- prefix Sum 的index和 sum存在字典里
- 遍历字典，并根据对应差找到对应字典中可能的index，寻找最大值

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        d = defaultdict(list)
        s = 0
        for i in range(len(nums)):
            d[s].append(i)
            s += nums[i]
        d[s].append(i+1)
        # print(d)
        res = 0
        for s, ls in d.items():
            if k + s in d: 
                res = max(res, d[k+s][-1] - ls[0])
        return res            
```

### [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

### [Range Addition](https://leetcode.com/problems/range-addition/)

```python
class Solution(object):
    def getModifiedArray(self, length, updates):
        """
        :type length: int
        :type updates: List[List[int]]
        :rtype: List[int]
        """
        res = [0] * length
        for s, e, diff in updates:
            res[s] += diff
            
            if e + 1 <= length - 1:
                res[e+1] -= diff
        sum = 0
        for i in range(length):
            sum += res[i]
            res[i] = sum
        return res
```