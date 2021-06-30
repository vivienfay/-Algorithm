###### 总结

1. 使用多个指针
    - 指针方向相反：two sum less than k
    - 指针方向相同
    - 使用swap的方法： 



2. 转化成set


3. subarray
    - 二分法
    - store prefix information in hash table
    - sliding window
   
   
4. 使用hash map记住index


5. slidingn window


6. #### interval题目:
    - 排序
    - greedy
    - 扫描线
    - DP

### 题型分类

- #### 先排序再做指针循环
    - [3Sum](#15)
    - [3Sum Closest](#16)
    - [4Sum](#18)
    - [3Sum Smaller](#259)
    
- #### 单指针
    - [Maximum Subarray](#53)
    - [Monotonic Array](#896)
    
    - [Increasing Triplet Subsequence](#334)

- #### 双指针
    - #### 同向
    - [Add Strings](#415)
    - [Add Binary](#67)
    - [Next Permutation](#34)
    - [Best Time to Buy and Sell Stock II](#122)
    - [Long Pressed Name](#925)
    - [Intersection of Two array](#349)
    - [Intersection of Three array]()

3. 快慢指针
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

- #### Presum/Prefix Product

    - [Maximum Subarray](#53)
    - [Subarray Sum Equals K](#560)
    - [Subarray Sums Divisible by K](#974)
    - [Product of the Last K Numbers](#1352)
    
- #### two direction

    - [Product of Array Except Self](#238)
    - [Trapping Rain Water](#42)
    
- #### swap
    - [Rotate Image](#48)
    - [Remove Duplicates from Sorted Array](#26)
    - [Remove Duplicates from Sorted Array II](#80)
    - [Move Zeros](#283)
    - [Shortest Word Distance III](#245)
    - [Minimum Swap](#670)
 
    
- #### Interval(greedy)
    - 排序后让可能重叠的interval相邻
    - [Merge Intervals](#56)
    - [Insert Interval](#57)
    - [Meeting Room](#252)
    - [Meeting Room II](#253) 
    - [Non-overlapping Intervals](#435)
    - [Find Right Interval](#436)
    - [Remove Interval](#1272)
    - [Data Stream as Disjoint Intervals](#352)
    - [Add Bold Tab in String](#616)
    - [Exclusive Time of Functions](#636)
    - [Falling Squares](#699)
    - [Range Module](#715)
    - [Employee Free Time](#759)
    - [Interval List Intersections](#986)
    
- #### Sliding Window
    - [Permutation in String](#567)
    - [Find All Anagrams in a String](#438)
    - [Longest Substring Without Repeating Characters](#3)
    - [Longest Substring with At Most K Distinct Characters](#340)
    - [Minimum Window Substring](#76)
    - [Longest Substring with At Most Two Distinct Characters](#159)
    - [Fruit Into Baskets](#504)
    - [Replace the Substring for Balanced String](#1234)
    - [Grumpy Bookstore Owner](#1052)
    - [Max Consecutive Ones III](#1004)
    - [Count Number of Nice Subarrys]
    - [Replace the Substring for Balaced String]
    - [Binary Subarrays with sum]
    - [](#424)
    - [Maximum Points You Can Obtain from Cards](#1423)
    - [Subarrays with k different integers](#992)
    - [Shortest subarray with sum at least K]
    - [minimum size subarray Sum](#209)
    
- #### 字符
    - [Group shifted Strings](#249)


- #### 记录特别的变量
    - [Best Time to Buy and Sell Stock](#121)
    - [Shortest Unsorted Continuous Subarray](#581)
    
- #### 三指针
    - [Intersection of Three Sorted Arrays](#1214)
    
- #### 用-1做标记
    - [Find All Duplicates in an Array](#442)
    - [Find All Numbers Disappeared in an Array](#448)
    
- #### hashmap
    - [Two Sum]()
    - [Number of Good Pairs](#1512)
    - [Partition Labels](#763)
    - [Longest Consecutive Sequence](#128)
    - [Group Anagram](#)

- #### Monotomic Queue
    - [Sliding Window Maximum](#239)

- #### Heap
    - [Top K frequent Word](#)
    - [Task Scheduler](#621)
    - [Reorganize String](#767)
    - [Minimum Cost to Connect Sticks](#1167)
    - [K Closest Points to Origin](#973)
    - [Find Median from Data Stream](#295)

- #### Greedy
    - [Queue Reconstruction by Height](#406)

### 易错点

### 必备模版技巧


# 1
### Two Sum

[Leetcode](https://leetcode.com/problems/two-sum/)
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

# 1099
### Two Sum Less Than K
[Leetcode](https://leetcode.com/problems/two-sum-less-than-k/)
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

# 334
### Increasing Triplet Subsequence
[Leetcode](https://leetcode.com/problems/increasing-triplet-subsequence/)


```python
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        first = second = float('inf')
        for i in nums:
            if i <= first:
                first = i
            elif i <= second:
                second = i
            else: return True
        return False
```

# 15
### Three Sum
[Leetcode](https://leetcode.com/problems/3sum/)
三个指针轮番走
注意unique，可以while循环remove 重复值


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

# 16
### 3Sum Closest
[Leetcode](https://leetcode.com/problems/3sum-closest/m)


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

# 18
### 4 Sum

- 注意duplicate的时候 用while移动指针


18.
https://leetcode.com/problems/4sum/submissions/


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

# 283
### Move Zeros
[Leetcode](https://leetcode.com/problems/move-zeroes/)
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

# 259
[Leetcode](https://leetcode.com/problems/3sum-smaller/)
### 3Sum Smaller


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

# 26
### Remove Duplicates from Sorted Array
[Leetcode](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
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

# 80
### Remove Duplicates from Sorted Array II
[Leetcode](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)



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

# 268

### Missing Number
[Leetcode](https://leetcode.com/problems/missing-number/)


```python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        new_set = set(nums)
        for i in range(n+1):
            if i not in new_set:
                return i
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

# 349
### intersection of two arrays
[Leetcode](https://leetcode.com/problems/intersection-of-two-arrays/)


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

### K-diff Pairs in an Array

532
https://leetcode.com/problems/k-diff-pairs-in-an-array/


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

# 57
### insert interval
[Leetcode](https://leetcode.com/problems/insert-interval/)

- 两种做法
1. 先把新的interval插入进去，再用和merge interval一样的算法 可以直接调用 O(n)/O(n) 
2. 找index


```python
# solution 1
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        start, end = newInterval
        ind = 0
        while ind < len(intervals) and intervals[ind][0] < start:
            ind += 1
        intervals = intervals[:ind] + [newInterval] + intervals[ind:]
        res = [intervals[0]]
        for i, j in intervals[1:]:
            start, end = res[-1]
            if i <= end: res[-1][1] = max(j, end)
            else: res.append([i, j])
        return res
         
        
        
```


```python
# solution 2
class Solution:
    def insert(self, intervals, newInterval):
        res, n = [], newInterval
        for index, i in enumerate(intervals):
            if i[1] < n[0]:
                res.append(i)
            elif n[1] < i[0]:
                res.append(n)
                return res + intervals[index:]
            else:
                n[0] = min(n[0],i[0])
                n[1] = max(n[1], i[1])
        res.append(n)
        return res
```

# 435
### Non-overlapping Intervals
[Leetcode](https://leetcode.com/problems/non-overlapping-intervals/)


```python
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        end, cnt = float('-inf'), 0
        for s, e in sorted(intervals, key=lambda x: x[1]):
            if s >= end: 
                end = e
            else: 
                cnt += 1
	return cnt
```

### partition array

### Sort letters by oven and even

### Sort letters by case

# 75
### Sort Colors
[Leetcode](https://leetcode.com/problems/sort-colors/)


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

# 11
### Container with most Water
[Leetcode](https://leetcode.com/problems/container-with-most-water/)


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

### Merge Sorted Array

- 两个指针是从后面开始遍历的
- 考虑如果一个走完了，另一个怎么办


88
https://leetcode.com/problems/merge-sorted-array/


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

# 252

[Leetcode](https://leetcode.com/problems/meeting-rooms/)

### Meeting Room

- 还可以用扫描线算法，使用list作为端点容器然后排序



```python
# 排序，判断相邻interval是否有重叠
# o(nlogn)/o(1)
class Solution(object):
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        ls = sorted(intervals, key = lambda x: x[0])
        for i,j in zip(ls[:-1], ls[1:]):
            if i[1] > j[0]: return False
        return True
```

# 253

[Leetcode](https://leetcode.com/problems/meeting-rooms-ii/)

### Meeting Room II

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

# 56

[Leetcode](https://leetcode.com/problems/merge-intervals/)

### Merge Intervals


```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals: return []
        intervals = sorted(intervals, key = lambda x: x[0])
        res = [intervals[0]]
        for i in intervals[1:]:
            end = res[-1][1]
            if i[0] <= end: res[-1][1] = max(i[1], end)
            else: res.append(i)
        return res
```

# 986

### Interval List Intersections
[Leetcode](https://leetcode.com/problems/interval-list-intersections/)


```python
class Solution:
    def intervalIntersection(self, A, B):
        res = []
        i, j = 0, 0
        while i < len(A) and j < len(B):
            s1, e1, s2, e2 = A[i][0], A[i][1], B[j][0], B[j][1]
            if e2 < s1: 
                j += 1
            elif e1 < s2: 
                i += 1
            else:
                start = max(s1, s2)
                end = min(e1,e2)
                res.append([start, end])
                if e1 > e2: 
                    j += 1
                else:
                    i += 1
        return res
```

# 1272
### Remove Interval
[Leetcode](https://leetcode.com/problems/remove-interval/)


```python
class Solution(object):
    def removeInterval(self, intervals, toBeRemoved):
        """
        :type intervals: List[List[int]]
        :type toBeRemoved: List[int]
        :rtype: List[List[int]]
        """
        res = []
        x, y = toBeRemoved
        for i, j in intervals:
            if j <= x or y <= i: res.append([i, j])
            else:
                if i < x: res.append([i, x])
                if j > y: res.append([y, j])
        return res
            
```

### Minimum Size Subarray Sum

- sliding window

209
https://leetcode.com/problems/minimum-size-subarray-sum/


```python
class Solution:
    def minSubArrayLen(self, s, nums):
        left = total = 0
        min_res = float('inf')
        for right, value in enumerate(nums):
            total += value
            while total >= s and left <= right:
                min_res = min(min_res, right - left + 1)
                total -= nums[left]
                left += 1
        return min_res if min_res <= len(nums) else 0
```

# 647
### Palindromic Substrings  

- 以index为中心往左右散开test是不是palindromic

[Leetcode](https://leetcode.com/problems/palindromic-substrings/)



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

### Longest Palindromic Substring

5.
https://leetcode.com/problems/longest-palindromic-substring/


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

# 53

[leetcode](https://leetcode.com/problems/maximum-subarray/)

### Maximum Subarray
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

# 560

[Leetcode](https://leetcode.com/problems/subarray-sum-equals-k/)

### Subarray Sum Equals K
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

# 974

[Leetcode](https://leetcode.com/problems/subarray-sums-divisible-by-k/)

### Subarray Sums Divisible by K

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

# 238

[Leetcode](https://leetcode.com/problems/product-of-array-except-self/)

### Product of Array Except Self



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

# 42

[Leetcode](https://leetcode.com/problems/trapping-rain-water/)

### Trapping Rain Water

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

# 48

[Leetcode](https://leetcode.com/problems/rotate-image/)

### Rotate Image




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

# 415

[Leetcode](https://leetcode.com/problems/add-strings/)

### Add Strings


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

# 67

[Leetcode](https://leetcode.com/problems/add-binary/)

### Add Binary


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

# 953

[Leetcode](https://leetcode.com/problems/verifying-an-alien-dictionary/)

### Verifying an Alien Dictionary


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

# 31
[Leetcode](https://leetcode.com/problems/next-permutation/)

### Next Permutation


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

# 54
[Leetcode](https://leetcode.com/problems/spiral-matrix/)

### Spiral Matrix


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

# 121

[Leetcode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)


### Best Time to Buy and Sell Stock



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

# 443
### String Compression
[Leetcode](https://leetcode.com/problems/string-compression/)


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

# 122
[Leetcode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

### Best Time to Buy and Sell Stock II
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

# 350

[Leetcode](https://leetcode.com/problems/intersection-of-two-arrays-ii/)
### Intersection of Two Arrays II


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

# 567
[Leetcode](https://leetcode.com/problems/permutation-in-string/)

### Permutation in String


```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        map = {}
        for i in s1:
            map[i] = map.get(i, 0) + 1
        count = len(s1)
        left = right = 0
        while right < len(s2):
            map[s2[right]] = map.get(s2[right], 0) - 1
            if map[s2[right]] >= 0: count -= 1
            right += 1
            while count == 0:
                if count == 0 and right - left == len(s1): return True
                map[s2[left]] += 1
                if map[s2[left]] >= 1: count += 1
                left += 1
        return False
```


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

# 438
[Leetcode](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

### Find All Anagrams in a String


```python
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        map = {}
        for i in p:
            map[i] = map.get(i, 0) + 1
        count = len(p)
        i = j = 0
        res = []
        while j < len(s):
            map[s[j]] = map.get(s[j], 0) - 1
            if map[s[j]] >= 0: count -= 1
            j += 1
            while count == 0:
                if count == 0 and len(p) == j - i: 
                    res.append(i)
                map[s[i]] = map[s[i]] + 1
                if map[s[i]] >= 1: count += 1
                i += 1        
        return res
```


```python
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        from collections import Counter
        d1 = Counter(p)
        d2 = {}
        i = j = 0
        length = 0
        res = []
        while j < len(s):
            ch = s[j]
            d2[ch] = d2.get(ch, 0) + 1
            length += 1
            if length == len(p):
                if d2 == d1: res.append(i)
                d2[s[i]] -= 1
                if d2[s[i]] == 0: del d2[s[i]]
                i += 1
                length -= 1
            j += 1
        return res
```

# 3
[Leetcode](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### Longest Substring Without Repeating Characters


```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        map = {}
        i = j = 0
        count = 0
        res = 0
        while j < len(s):
            map[s[j]] = map.get(s[j], 0) + 1
            if map[s[j]] > 1: count += 1
            j += 1
            if count == 0: res = max(res, j - i)
            while count > 0:
                map[s[i]] -= 1
                if map[s[i]] == 1: count -= 1
                i += 1
        return res
```

# 340

[Leetcode](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
### Longest Substring with At Most K Distinct Characters


```python
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        map = {}
        count = 0
        i = j = 0
        res = 0
        while j < len(s):
            map[s[j]] = map.get(s[j], 0) + 1
            if map[s[j]] == 1: count += 1
            j += 1
            if count <= k: res = max(res, j - i)
            while count > k:
                map[s[i]] -= 1
                if map[s[i]] == 0: count -= 1
                i += 1
        return res
```

# 1151
### Minimum Swaps to Group All 1's Together
[Leetcode](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/)


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

# 76
[Leetcode](https://leetcode.com/problems/minimum-window-substring/)

###  Minimum Window Substring


```python
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        map = {}
        for i in t:
            map[i] = map.get(i, 0) + 1
        i = j = 0
        count = len(t)
        rescount = len(s)
        res = ''
        while j < len(s):
            map[s[j]] = map.get(s[j], 0) - 1
            if map[s[j]] >= 0: count -= 1
            j += 1
            while count == 0: 
                if j - i <= rescount:
                    rescount = j - i
                    res = s[i:j]
                map[s[i]] += 1
                if map[s[i]] >= 1: count += 1
                i += 1
        return res
```

# 159

[Leetcode](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/)
###  Longest Substring with At Most Two Distinct Characters


```python
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        map = {}
        i = j = 0
        count = 0
        res = 0
        while j < len(s):
            map[s[j]] = map.get(s[j], 0) + 1
            if map[s[j]] == 1: count += 1
            j += 1
            if count <= 2: res = max(res, j - i)
            while count > 2: 
                map[s[i]] -= 1
                if map[s[i]] == 0: count -= 1
                i += 1
        return res
```

# 904

[Leetcode](https://leetcode.com/problems/fruit-into-baskets/)
### Fruit Into Baskets


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

# 1234
[Leetcode](https://leetcode.com/problems/replace-the-substring-for-balanced-string/)

### Replace the Substring for Balanced String

- 记录窗口外的情况


```python
class Solution(object):
    def balancedString(self, s):
        """
        :type s: str
        :rtype: int
        """
        count = collections.Counter(s)
        res = n = len(s)
        i = 0
        for j, c in enumerate(s):
            count[c] -= 1
            while i < n and all(n / 4 >= count[c] for c in 'QWER'):
                res = min(res, j - i + 1)
                count[s[i]] += 1
                i += 1
        return res  
```

# 1156
### Swap For Longest Repeated Character Substring
[Leetcode](https://leetcode.com/problems/swap-for-longest-repeated-character-substring/)

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

# 424
### Longest Repeating Character Replacement
[Leetcode](https://leetcode.com/problems/longest-repeating-character-replacement/)


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

# 1100
### Find K-Length Substrings With No Repeated Characters
[Leetcode](https://leetcode.com/problems/find-k-length-substrings-with-no-repeated-characters/)


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

# 992
### Subarrays with K Different Integers
[Leetcode](https://leetcode.com/problems/subarrays-with-k-different-integers/)

1. exact k = (at most k) - (at most k-1)
2. count += j - i，j的指针已经预先往后了一个




```python

# 0  1  2  3  4  5
# i:0 j : 3
# 0,1,2,3
# 1,2,3
# 2,3
# 3

```


```python
class Solution(object):
    def subarraysWithKDistinct(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # return self.helper(A, K) - self.helper(A, K-1) 
        return self.helper(A, K) - self.helper(A, K - 1)
        
    def helper(self, A, K):
        count = 0
        i = j = 0
        d = {}
        while j < len(A):
            d[A[j]] = d.get(A[j], 0) + 1
            j += 1
            while i < j and len(d) > K:
                d[A[i]] -= 1
                if d[A[i]] == 0: del d[A[i]]
                i += 1
            count += j - i
        return count
```

# 1052
[Leetcode](https://leetcode.com/problems/grumpy-bookstore-owner/)
### Grumpy Bookstore Owner


```python
class Solution(object):
    def maxSatisfied(self, customers, grumpy, X):
        """
        :type customers: List[int]
        :type grumpy: List[int]
        :type X: int
        :rtype: int
        """
        total = 0
        for i, j in zip(customers, grumpy):
            if j == 0: total += i
        i = j = 0
        res =  0
        while j < len(customers):
            if grumpy[j] == 1: total += customers[j]
            j += 1
            res = max(res, total)
            if j - i >= X:
                if grumpy[i] == 1: total -= customers[i]
                i += 1
        return res

```

# 1004
[Leetcode](https://leetcode.com/problems/max-consecutive-ones-iii/)
### Max Consecutive Ones III


```python
class Solution(object):
    def longestOnes(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        i = j = 0
        count = 0
        res = 0
        while j < len(A):
            if A[j] == 0: count += 1
            j += 1
            if count <= K: res = max(res, j - i)
            while count > K:
                if A[i] == 0: count -= 1
                i += 1
        return res
```

# 925
[Leetcode](https://leetcode.com/problems/long-pressed-name/)

### Long Pressed Name


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

# 1213
[Leetcode](https://leetcode.com/problems/intersection-of-three-sorted-arrays/)
### Intersection of Three Sorted Arrays


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

# 442
[Leetcode](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
### Find All Duplicates in an Array

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

# 448
### Find All Numbers Disappeared in an Array
[Leetcode](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)


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

# 55
[Leetcode](https://leetcode.com/problems/jump-game/)
### Jump Game


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

# 125
[Leetcode](https://leetcode.com/problems/valid-palindrome/)
# Valid Palindrome


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

# 896
### Monotonic Array
[Leetcode](https://leetcode.com/problems/monotonic-array/)


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

# 1512
### Number of Good Pairs
[Leetcode](https://leetcode.com/problems/number-of-good-pairs/)


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

# 763
### Partition Labels
[Leetcode](https://leetcode.com/problems/partition-labels/)


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

# 1041
### Robot Bounded In Circle
[Leetcode](https://leetcode.com/problems/robot-bounded-in-circle/)


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

# 1658
### Minimum Operations to Reduce X to Zero
[Leetcode](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)


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

# 128
### Longest Consecutive Sequence
[Leetcode](https://leetcode.com/problems/longest-consecutive-sequence/)


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

# 977
### Squares of a Sorted Array
[Leetcode](https://leetcode.com/problems/squares-of-a-sorted-array/)


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

# 680
### Valid Palindrome II 
[Leetcode](https://leetcode.com/problems/valid-palindrome-ii/)


```python
class Solution(object):
    def validPalindrome(self, s):
        i = 0
        while i < len(s) / 2 and s[i] == s[-(i + 1)]: i += 1
        s = s[i:len(s) - i]
        return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]
```

# 347
### Top K Frequent Elements
[Leetcode](https://leetcode.com/problems/top-k-frequent-elements/)


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

# 49
### Group Anagrams
[Leetcode](https://leetcode.com/problems/group-anagrams/)


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

# 249
### Group Shifted mStrings
[Leetcode](https://leetcode.com/problems/group-shifted-strings/)


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

# 406
### Queue Reconstruction by Height
[Leetcode](https://leetcode.com/problems/queue-reconstruction-by-height/)


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

# 215
### Kth Largest Element in an Array
[Leetcode](https://leetcode.com/problems/kth-largest-element-in-an-array/)


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

# 581
### Shortest Unsorted Continuous Subarray
[Leetcode](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)



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

# 1423
### Maximum Points You Can Obtain from Cards
[Leetcode](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)


```python
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        res = float('inf')
        count = 0
        i = j = 0
        while j < len(cardPoints):
            count += cardPoints[j]
            j += 1
            if j - i > len(cardPoints) - k:
                count -= cardPoints[i]
                i += 1
            if j - i == len(cardPoints) - k: res = min(res, count)
        return sum(cardPoints) - res  
        
            
        
```

# 209
### Minimum Size Subarray Sum
[Leetcode](https://leetcode.com/problems/minimum-size-subarray-sum/)


```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        i, res, total = 0, len(nums) + 1, 0
        for j in xrange(len(nums)):
            total += nums[j]
            while total >= s:
                res = min(res, j - i + 1)
                total -= nums[i]
                i += 1
        return res % (len(nums) + 1)
                
```

# 881
### Boats to Save People
[Leetcode](https://leetcode.com/problems/boats-to-save-people/)


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

# 36
### Valid Sudoku
[Leetcode](https://leetcode.com/problems/valid-sudoku/)


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

# 239
### Sliding Window Maximum
[Leecode](https://leetcode.com/problems/sliding-window-maximum/)


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

# 767
### Reorganize String
[Leetcode](https://leetcode.com/problems/reorganize-string/)

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

# 621
### Task Scheduler
[Leetcode](https://leetcode.com/problems/task-scheduler/)


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

# 670
### Maximum Swap
[Leetcode](https://leetcode.com/problems/maximum-swap/)


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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-31-a5b5fae0872a> in <module>
    ----> 1 A = map(int, str(num))
          2 last = {x: i for i, x in enumerate(A)}
          3 for i, x in enumerate(A):
          4     for d in xrange(9, x, -1):
          5         if last.get(d, None) > i:


    NameError: name 'num' is not defined


# 1167
### Minimum Cost to Connect Sticks
[Leetcode](https://leetcode.com/problems/minimum-cost-to-connect-sticks/)


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

# 973
### K Closest Points to Origin
[Leetcode](https://leetcode.com/problems/k-closest-points-to-origin/)


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

# 289
### Game of Life 
[Leetcode](https://leetcode.com/problems/game-of-life/)


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

# 259
### Find Median from Data Stream
[Leetcode](https://leetcode.com/problems/find-median-from-data-stream/)


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

# 1352
### Product of the Last K Numbers
[Leetcode](https://leetcode.com/problems/product-of-the-last-k-numbers/)

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


```python

```
