# Sliding Window 总结

## 总结

- 什么时候会用到sliding window
  - 求 longest subarray
  - 求 shortest subarray
  - 如果有pattern permutation(其实实质上就是subarray)，可以考虑把pettern化作hashmap 找对应是的cnt map是否在string中存在
  - 如果是要求连续，很有可能
- 时间复杂度通常是o（n）， 空间复杂度中，如果有使用hashmap存储 应该是o（k）
- 解题注意点
  - 沟通的时候可以说，每次iterate一个letter，然后遇到什么情况下才需要shrink window
- 解题习惯与技巧
  - 可以使用for j in range() 然后再嵌套while 比较不容易出错


## 题型分类

- [Sliding Window 总结](#sliding-window-总结)
  - [总结](#总结)
  - [题型分类](#题型分类)
    - [Hard: Minimum Window Substring](#hard-minimum-window-substring)
    - [Hard: Permutation in String](#hard-permutation-in-string)
    - [Find All Anagrams in a String](#find-all-anagrams-in-a-string)
    - [Longest Substring Without Repeating Characters](#longest-substring-without-repeating-characters)
    - [Longest Substring with At Most K Distinct Characters](#longest-substring-with-at-most-k-distinct-characters)
    - [Longest Substring with At Most Two Distinct Characters](#longest-substring-with-at-most-two-distinct-characters)
    - [Replace the Substring for Balanced String](#replace-the-substring-for-balanced-string)
    - [Grumpy Bookstore Owner](#grumpy-bookstore-owner)
    - [Max Consecutive Ones III](#max-consecutive-ones-iii)
    - [Maximum Points You Can Obtain from Cards](#maximum-points-you-can-obtain-from-cards)
    - [Subarrays with K Different Integers](#subarrays-with-k-different-integers)
    - [Minimum Size Subarray Sum](#minimum-size-subarray-sum)
    - [Fruit Into Baskets](#fruit-into-baskets)
    - [Longest Repeating Character Replacement](#longest-repeating-character-replacement)
    - [Max Consecutive Ones III](#max-consecutive-ones-iii-1)
    - [Substring with Concatenation of All Words](#substring-with-concatenation-of-all-words)

------

### [Hard: Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

维护一个dictionary，key为字母，value为字母的count。初始化时是t字符串的字母个数，随着指针移动，字典中的count代表了区间范围内还剩多少个字母需要包含才能符合条件。同时cnt表示还剩下多少个数字，当达到零，说明区间内已经达到要求，可以进行清算，去更新min_range。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import Counter
        d = Counter(t)
        cnt = len(t)
        res = len(s) + 1
        ans = ''
        l, r = 0, 0
        while r < len(s):
            d[s[r]] = d.get(s[r], 0) - 1
            if d[s[r]] >= 0: cnt -= 1
            r += 1
            while cnt == 0:
                if res >= r - l:
                    res = r - l
                    ans = s[l:r]
                d[s[l]] += 1
                if d[s[l]] > 0: cnt += 1
                l += 1
        return ans
```

### [Hard: Permutation in String](https://leetcode.com/problems/permutation-in-string/)

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

### [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

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

### [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

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

### [Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)

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

### [Longest Substring with At Most Two Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/)

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

### [Replace the Substring for Balanced String](https://leetcode.com/problems/replace-the-substring-for-balanced-string/)

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

### [Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/)

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

### [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)

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

### [Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)

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

### [Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/)

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

### [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

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

### [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)

```python
class Solution(object):
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        d = {}
        i = 0
        res = 0
        for j in range(len(fruits)):
            d[fruits[j]] = d.get(fruits[j], 0) + 1
            while len(d) > 2:
                d[fruits[i]] -= 1
                if d[fruits[i]] == 0: del d[fruits[i]]
                i += 1
            res = max(res, j - i + 1)  
        return res
```

### [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

- We will iterate through the string to add one letter at a time in the window.
- We will also keep track of the count of the maximum repeating letter in any window.
- Why don't we update the maxcurCnt? since we have to replace all the remaining letters to get the longeest substring having the same letter in any window,

```python
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        i = 0
        res = 0
        d = {}
        curCnt = 0
        for j in range(len(s)):
            d[s[j]] = d.get(s[j], 0) + 1
            curCnt = max(d[s[j]], curCnt)
            if j - i + 1 - curCnt > k:
                d[s[i]] -= 1
                i += 1
            res = max(res, j - i + 1)
        return res

```

### [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)

```python
def length_of_longest_substring(arr, k):
  # TODO: Write your code here
  i = 0
  res = 0
  cnt0 = 0
  for j in range(len(arr)):
    if arr[j] == 0: cnt0 += 1
    while cnt0 > k:
      if arr[i] == 0: cnt0 -= 1
      i += 1
    res = max(res, j - i + 1)

  return res
```


### [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

- Keep the frequency of every word in a HashMap.
- Starting from every index in the string, try to match all the words.
- In each iteration, keep track of all the words that we have already seen in another HashMap.
- If a word is not found or has a higher frequency than required, we can move on to the next character in the string.
- Store the index if we have found all the words.

```python
def find_word_concatenation(str, words):
  from collections import Counter
  if len(words) == 0 or len(words[0]) == 0:
    return []
  d = {}
  word_d = Counter(words)
  res = []
  cnt = len(words)
  length = len(words[0])
  for i in range((len(str) - cnt * length)+1):
    seen = {}
    for j in range(0, cnt):
      next_word = i + j * length
      word = str[next_word: next_word + length]
      if word not in word_d:
        break
      seen[word] = seen.get(word, 0) + 1
      if seen[word] > word_d.get(word, 0): break
      if j + 1 == cnt:  # Store index if we have found all the words
        res.append(i)
  return res
```