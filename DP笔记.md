### 总结

#### 可考虑dp：
- 求最大值最小值
    - 从左上走到右下路径的最大数字和
    - 最长上升子序列长度
- 判断是否可行
    - 取石子游戏，先手是否必胜
    - 能不能选出k个数使得sum是k
- 统计方案个数
    - 有几种方式走到右下角
    - 有多少方法选出k个数使得和是sum

#### 极不可能用dp：
- 求出具体方案而非方案个数
- 输入数据是一个集合而不是序列
- 暴力算法的复杂度是多项式级别

#### 动态规划组成部分：
1. 确定状态
    - 最后一步
    - 化成子问题
2. 转移方程

    - f(x) = min(f(x-5), f(x-7))
3. 初始条件和边界情况
    - f(0) = 0
4. 计算顺序
    - f（0），f（1）


- 坐标型: 初始条件i[0]就是指以a0为结尾的子序列的性质
- 序列型：f[i]表示前i个元素，并随时记录状态
- 划分型：加上段数信息，记录j段性质
- 博弈：从第一步开始思考
        

#### 残酷算法
1. 时间序列型： 给出一个序列，其中每个元素可以认为一天，并且今天的状态只取决于昨天的状态。
    套路：
        - 定义dp[i][j]: 表示第i-th轮的第j种状态
        - 千方百计将dp[i][j]与前一轮的状态产生关系
        - 最终的结果是dp[last][j]中的某种aggregation
        
        
2. 时间序列加强版：给出一个序列，其中每个元素可以认为一天，并且今天的状态与之前的某一天有关。
    套路：
        - 定义dp[i] 状态和元素i直接有关
        - 千方百计将dp[i][j]与前一轮的状态产生关系
        - 最终的结果是dp[i]中的某一个
    一般两层循环
    
    
3. 双序列型： 给两个序列s和t
    套路：
        - 定义dp[i][j]:表示针对s[0:i] 和t[1:j]的子问题的求解
        - 千方百计把dp[i][j]往之前的状态去转移
        - 最终结果是dp[m][n]
        
4. 第i类区间型： 明确要求分割成k个连续区间，要你计算这些区间某个最优性质
    套路：
        - 状态定义：dp[i][k]表示针对s[1:i]分成k个区间，此时能够得到的最优解
        - 搜寻最后一个区间的起始位置j，将dp[i][k] 分割成dp[j-1][k-1]和s[j:i]两个部分
        - 最终的结果是dp[N][K]

#### 什么时候需要建立n+1大小的dp列？


### 题型分类

- #### 坐标型
    - [Unique Path](#62)
    - [Unique Path II](#63)
    - [Minimum Path sum](#64)
    - [Coin Change](#322)
    - [Coin Change II](#518) 
    - [Maximal Squares](#221)

- #### 序列型
    - [Jump Game](#55)
    - [Maximum Product Subarray](#152)
    - [Paint House](#256)
    - [Paint House II](#)
    - [Climbing Stairs](#70)
    - [House Robbery](#198)
    - [House Robbery II](#213)
    - [Counting Bits](#338)
    
- #### 划分型
    - [Decode ways](#91)
    - [Perfect Sqaures](#279)
    - [Palindrome Partition II]()
    - [Copy Books]()
    - [Integer break](#343)

- #### 位操作型
- #### 区间型
    - [Range Sum Query - Immutable](#303)
    - [Arithmetic Slices](#413)
    
    - [Maximum Sum of Two Non-Overlapping Subarrays](#1031)
- #### 背包型
    - [Backpack]()
    - [BackpackII]()
    - [Backpack III]()
    - [Target Sum](#494)



- #### 最长序列
    - [Longest Increasing Subsequence](#300)
    - [Maximum Length of Pair Chain](#646)
    - [Russian Doll Envelops](#354)
    
    
- #### 博弈型
    - [coins in a linne]()



- 综合型


- #### 字符匹配
    - [Edit Distance](#edit-distance)
    - [Wildcard Matching](#wildcard-matching)

### 易错点

### 必备模版/技巧

### - 坐标型

### Triangle

### minimum path sum

# 62
[Leetcode](https://leetcode.com/problems/unique-paths/)
### unique path


```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0] * n] * m
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0: 
                    dp[i][j] = 1
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]                
```

# 63

[Leetcode](https://leetcode.com/problems/unique-paths-ii/)
### Unique path II


```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1: break
            dp[i][0] = 1
        for i in range(n):
            if obstacleGrid[0][i] == 1: break
            print(dp, i)
            dp[0][i] = 1
            
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

### knight shortest path

# 70
[Leetcode](https://leetcode.com/problems/climbing-stairs/)
### climbing stairs


```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0 for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = 1
        if n >= 2:
            dp[2] = 2
            for i in range(3, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]
```


```python
# 优化空间做法(滚动数组变形-滚动元素)
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        pre = 0
        cur = 1
        if n >= 2:
            pre, cur = cur, 2
            for i in range(3, n + 1):
                pre, cur = cur, pre + cur
        return cur
```

### 跳跃游戏

### - 接龙型

# 279

[Leetcode](https://leetcode.com/problems/perfect-squares/)
### Perfect Squares


```python
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [float('inf') for _ in range(n + 1)]
        dp[0] = 0
        for i in range(1, n + 1):
            j = 0
            while i - j * j >= 0:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        return dp[-1]
```

# 322
[Leetcode](https://leetcode.com/problems/coin-change/)
### Coin Change


```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        m = len(coins)
        dp = [float('inf') for _ in range(amount + 1)]
        dp[0] = 0
        for i in range(m):
            j = 0
            while j + coins[i] <= amount:
                if dp[j] != float('inf'):
                    dp[j + coins[i]] = min(dp[j + coins[i]], dp[j] + 1)
                j += 1
        return dp[amount] if dp[amount] != float('inf') else -1
```

# 518
[Leetcode](https://leetcode.com/problems/coin-change/)
### Coin Change ii


```python
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        n = len(coins)
        dp = [0 for _ in range(amount + 1)]
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i-coin]
        return dp[amount]
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

# 152
### Maximum Product Subarray


```python
class Solution(object):
    def maxProduct(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(A)
        f = [0] * n
        g = [0] * n
        res = -float('inf')
        for i in range(n):
            f[i], g[i] = A[i], A[i]
            if i > 0: f[i] = max(f[i], max(f[i-1] * A[i], g[i-1] * A[i]))
            if i > 0: g[i] = min(g[i], min(f[i-1] * A[i], g[i-1] * A[i]))
            res = max(res,f[i])
        return res
```

# 91
[Leetcode](https://leetcode.com/problems/decode-ways/)
### Decode Ways


```python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:return 0
        dp = [0] * (len(s) + 1)
        dp[0] = 1 
        dp[1] = 0 if s[0] == "0" else 1 
        for i in range(2, len(s) + 1): 
            if 0 < int(s[i-1:i]) <= 9: dp[i] += dp[i - 1]
            if 10 <= int(s[i-2:i]) <= 26: dp[i] += dp[i - 2]
        return dp[len(s)]
```

# 300

[Leetcode](https://leetcode.com/problems/longest-increasing-subsequence/)
### Longest Increasing Subsequence


```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return 0
        dp = [1] * (n)
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```


```python
# 优化时间复杂度o（nlogn）
class Solution:
    def lengthOfLIS(self, nums):
        arr = []
        for i in nums:
            if not arr: arr.append(i)
            else:
                l, r = 0, len(arr) - 1
                while l + 1 < r:
                    mid = (l + r) // 2
                    if arr[mid] >= i: r = mid
                    else: l = mid
                if arr[l] >= i: arr[l] = i
                elif arr[r] >= i: arr[r] = i
                else: arr.append(i)  
        return len(arr)                        
```

# 646
[Leetcode](https://leetcode.com/problems/maximum-length-of-pair-chain/)
### Maximum Length of Pair Chain


```python
class Solution(object):
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        n = len(pairs)
        if n == 0: return 0
        pairs = sorted(pairs, key = lambda x: (x[0], x[1]))
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                a, b = pairs[j]
                c, d = pairs[i]
                if b < c:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
        
```


```python
# greedy o（nlogn）
class Solution(object):
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        N = len(pairs)
        pairs.sort(key = lambda x: x[1])
        ans = 0
        cur = -float('inf')
        for head, tail in pairs:
            if head > cur:
                cur = tail
                ans += 1
        return ans
```


```python
# 二分
class Solution(object):
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        pairs.sort()

        # min_end_y[i] is the ending tuple minimum y of length=i chain
        min_end_y = [float('inf')] * len(pairs)
        print(pairs)
        for x, y in pairs:
            # since (a, b) can chain (c, d) iff b < c, use bisect_left
            i = bisect.bisect_left(min_end_y, x)
            print(min_end_y)
            print(i)
            # greedy method, for the same length chain, the smaller y the better
            min_end_y[i] = min(min_end_y[i], y)  

        return bisect.bisect_left(min_end_y, float('inf'))
```


```python

```

# 64
[Leetcode](https://leetcode.com/problems/minimum-path-sum/)
### Minimum Path Sum


```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        if m != 0: n = len(grid[0])
        dp = [[0] * n] * m
        for i in range(m):
            for j in range(n):
                if i >= 1 and j >= 1:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
                elif j < 1:  dp[i][j] = dp[i-1][j] + grid[i][j]
                elif i < 1:  dp[i][j] = dp[i-1][j] + grid[i][j]
        return dp[m-1][n-1]
```

- 滚动数组， 优化空间

# 

### Bomb Enemey


```python

```

#
### counting bitd


```python

```

# 256
[Leetcode](https://leetcode.com/problems/paint-house/)
### Paint Houst


```python
class Solution(object):
    def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        n = len(costs)
        if n == 0: return 0
        dp = [[float('inf') for _ in range(3)] for _ in range(n+1) ] 
        dp[0] = [0, 0, 0]
        for i in range(1,n+1):
            for j in range(3):
                for k in range(3):
                    if j != k: dp[i][j] = min(dp[i][j], dp[i-1][k] + costs[i-1][j])
        return min(dp[-1][0], dp[-1][1], dp[-1][2])
```

# 

### Paint House ii
-优化： 除去

# 198
[Leetcode](https://leetcode.com/problems/house-robber/)
### House Robbery


```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return 0
        dp = [-float('inf') for _ in range(n + 1)]
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, n + 1):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i - 1])
        return dp[-1]
        
```


```python
# 优化空间
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return 0
        pre = 0
        cur = nums[0]
        for i in range(2, n + 1):
            pre, cur = cur, max(cur, pre + nums[i - 1])
        return cur
```

# 213
[Leetcode](https://leetcode.com/problems/house-robber-ii/)
### House Robbery II
- 环形街区， 可以分成两个例子再做比较


```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def sim_rob(nums, i, j):
            prev, cur = 0, 0
            for ind in range(i, j):
                prev, cur = cur, max(cur, prev + nums[ind - 1])
            return cur
    
        if not nums: return 0
        elif len(nums) == 1: return nums[0]
        else:
            n = len(nums)
            return max(sim_rob(nums,1, n), sim_rob(nums,0, n-1))
```

# 354
[Leetcode](https://leetcode.com/problems/russian-doll-envelopes/)
### Russain Doll Envelopes

- 信封的一开始需要排序， 同时排序的顺序需要按照(x[0], -x[1])， 可以保证后面的那个可以覆盖掉大的。


```python
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        envelopes = sorted(envelopes, key = lambda x: (x[0], -x[1]))
        res = [[0,0]] * len(envelopes)
        size = 0
        for envelop in envelopes:
            i, j = 0, size
            while i < j:
                mid = (i + j) / 2
                if res[mid][0] < envelop[0] and res[mid][1] < envelop[1]: i = mid + 1
                else: j = mid
            res[i] = envelop
            size = max(size, i + 1)
        return size
```

# 303
[Leetcode](https://leetcode.com/problems/range-sum-query-immutable/)
### Range Sum Query - Immutable


```python
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        sum = 0
        self.sum_ls = [sum]
        for i in nums:
            sum += i
            self.sum_ls.append(sum)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sum_ls[j + 1] - self.sum_ls[i]
```

# 413
[Leetcode](https://leetcode.com/problems/arithmetic-slices/)
### Arithmetic Slices



```python
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        n = len(A)
        if n < 3: return 0
        dp = [0 for _ in range(n)]
        for i in range(2, n):
            if (A[i] - A[i-1]) == (A[i-1] - A[i-2]): dp[i] = 1 + dp[i-1]
        return sum(dp)
```

best time to buy and sell stock

russian doll envelop

# 343
[Leetcode](https://leetcode.com/problems/integer-break/)
### Integer break


```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0 for _ in range(n + 1)]
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], max(j * dp[i-j], j * (i-j)))
        return dp[-1]
```

# 583
[Leetcode](https://leetcode.com/problems/delete-operation-for-two-strings/)
### Delete Operation for two strings
-转换为最长序列


```python
class Solution(object):
    def minDistance(self, w1, w2):
            m, n = len(w1), len(w2)
            dp = [[0] * (n + 1) for i in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], dp[i][j] + (w1[i] == w2[j]))
            return m + n - 2 * dp[m][n]
```

### [Edit Distance](https://leetcode.com/problems/edit-distance/)

```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word2), len(word1)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            dp[0][i] = dp[0][i-1] + 1
        for j in range(1, m + 1):
            dp[j][0] = dp[j - 1][0] + 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word2[i-1] == word1[j-1]:
                    dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i-1][j-1])
                else:
                    dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i-1][j-1] + 1)
        return dp[-1][-1]
```

### [Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        n, m =len(s), len(p)
        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(1, m + 1):
            if p[i-1] == '*':
                dp[i][0] = dp[i-1][0]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[j-1] == p[i-1] or p[i-1] == '?':
                    dp[i][j] = dp[i-1][j-1]
                else:
                    if p[i-1] == '*':
                        dp[i][j] = dp[i-1][j-1] or dp[i-1][j] or dp[i][j-1]
        return dp[m][n]
```

# 746
### Min Cost Climbing Stairs
[Leetcode](https://leetcode.com/problems/min-cost-climbing-stairs/)


```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        dp = [0] * (len(cost))
        dp[0] = cost[0]
        if len(cost) >= 2:
            dp[1] = cost[1]
            for i in range(2, len(cost)):
                dp[i] = min(dp[i-1], dp[i-2])+cost[i]
        return min(dp[-1],dp[-2])
        
```

# 221
### Maximal Square
[Leetcode](https://leetcode.com/problems/maximal-square/)


```python
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i >= 1 and j >= 1: dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
                    else: dp[i][j] = 1
                    res = max(res, dp[i][j] ** 2)
                else: dp[i][j] = 0
        return res
        
```

# 338
### Counting Bits
[Leetcode](https://leetcode.com/problems/counting-bits/)


```python
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        
        res = [0] * (num+1)
        if num == 0: return [0]
        res[1] = 1
        for i in range(2, num+1):
            res[i] = res[i/2] + i%2
        return res
```

# 718
### Maximum Length of Repeated Subarray
[Leetcode](https://leetcode.com/problems/maximum-length-of-repeated-subarray/submissions/)


```python
class Solution(object):
    def findLength(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        dp = [[0 for _ in range(len(A) + 1)] for _ in range(len(B) + 1)]
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
        return max(max(row) for row in dp)

```

# 1031
### Maximum Sum of Two Non-Overlapping Subarrays
[Leetcode](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

1. O(n)/O(n) 
记录两个list， 一个记录l长度的最大dp，另一个记录m
每次做比较的时候把当前l长度和最大m长度 或者vice versa 的记录进去
2. 0（n） / o（1）
优化空间，


解析：https://zhuanlan.zhihu.com/p/102955681


```python
class Solution(object):
    def maxSumTwoNoOverlap(self, A, L, M):
        """
        :type A: List[int]
        :type L: int
        :type M: int
        :rtype: int
        """
        
        n = len(A)
        dpl = [0] * (n + 1)
        dpm = [0] * (n + 1)
        sum = [0] * (n + 1)
        for i in range(n):
            sum[i+1] = sum[i] + A[i]
        dpl[L] = sum[L]
        dpm[M] = sum[M]
        res = 0
        for i in range(n):
            if i >= L:
                dpl[i+1] = max(dpl[i], sum[i+1] - sum[i+1-L])
                res = max(res, sum[i+1] - sum[i+1-L] + dpm[i+1-L])
            if i >= M:
                dpm[i+1] = max(dpm[i], sum[i+1] - sum[i+1-M])
                res = max(res, sum[i+1] - sum[i+1-M] + dpl[i+1-M])
        return res

```


```python
for i in xrange(1, len(A)):
            A[i] += A[i - 1]
            print(A)
        res, Lmax, Mmax = A[L + M - 1], A[L - 1], A[M - 1]
        for i in xrange(L + M, len(A)):
            Lmax = max(Lmax, A[i - M] - A[i - L - M])
            Mmax = max(Mmax, A[i - L] - A[i - L - M])
            res = max(res, Lmax + A[i] - A[i - M], Mmax + A[i] - A[i - L])
        return res
```


      File "<tokenize>", line 4
        res, Lmax, Mmax = A[L + M - 1], A[L - 1], A[M - 1]
        ^
    IndentationError: unindent does not match any outer indentation level



# 935
### Knight Dialer
[Leetcode](https://leetcode.com/problems/knight-dialer/)


```python
class Solution(object):
    def knightDialer(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [[0] * 10 for _ in range(n)]
#       define the path
        m = {1:[6, 8], 2: [7, 9], 3: [4, 8], 4: [3, 9, 0], 6: [1, 7, 0],
            7: [2, 6], 8: [1, 3], 9: [2, 4], 0: [4, 6]}
#         initialize
        for i in range(10):
            dp[0][i] = 1
        for i in range(1, n):
            for j in range(10):
                if j in m:
                    for z in m[j]:
                        dp[i][j] += dp[i-1][z]
        return sum(dp[-1]) % (10 ** 9 + 7)
```