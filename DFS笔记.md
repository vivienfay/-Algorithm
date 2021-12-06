### 总结


- 和排列组合相关的题目
- 不是排列 就是组合
- 复杂度计算： o（答案个数 * 构造每个答案的时间复杂度）
- 非递归怎么办： 必背程序

- 注意backtracking的用法 


必背 ： tree treaversal / combination / permutation

### 题型分类


- #### Backtracking
    
    - [Letter Combinations of a Phone Number](#17)
    - [Generate Parentheses](#22)
    - [Combination Sum](#39)
    - [Combination Sum II](#40)
    - [Combination](#77)
    - [Permutation](#46)
    - [Permutation II](#47)
    - [Subsets](#78)
    - [Subsets II](#90)
    - [Palindrome Parititioning](#131)
    - [Word Search](#79)
    - [The k-th Lexicographical String of All Happy Strings of Length n](#1415)
    - [n-Queens](#51)
    - [n-Queens II](#52)
    - [Target Sum](#494)
    - [Max Area of Island](#695)
    - [Beautiful Arrangement](#526)
    - [Restore IP Addresses](#93)
    - [Maximum Length of a Concatenated String with Unique Characters](#1239)
    
- #### 排列组合型

- #### Memotization
    - [Longest Increasing Path in a Matrix](#329)
    - [Word Break](#139)
    - [](#)

- #### 循环嵌套dfs
    - [Surrounded Regions](#130)

- #### 图
    - [All Paths from Source Lead to Destination](#1059)
    - [Number of Distinct Islands](#694)
    - [Reconstruct Itinerary](#332)
    - [All Paths From Source to Target](#797)
    - [Is Graph Bipartite?](#785)
    
- #### dfs/bfs同时使用
    - [Shortest Bridge](#934)
    - [All Nodes Distance K in Binary Tree](#863)
    - [Concanated Words](#472)

### 易错点

### 必备模版技巧

----

# 39

[Leetcode](https://leetcode.com/problems/combination-sum/submissions/)
### Combination Sum

- 注意list copy的时候



```python
class Solution(object):
    def combinationSum(self, candidates, target):
        def dfs(cur, sid):
            if sum(cur) == target: 
                res.append(cur)
                return
            for i in range(sid, len(candidates)):
                if sum(cur) > target: break
                dfs(cur + [candidates[i]], i)
                
        if len(candidates) == 0: return []
        res = []
        candidates.sort()
        dfs([], 0)
        return res
```

# 40

[https://leetcode.com/problems/combination-sum/submissions/]

### Combination Sum ii

- 注意list copy的时候



```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(pos, subset, cursum):
            if target < cursum: return
            if target == cursum: 
                res.append(subset)
                return
            for i in range(pos, len(candidates)):
                if  i > pos and candidates[i] == candidates[i-1]: continue
                dfs(i + 1, subset + [candidates[i]], cursum + candidates[i])
                
            
        candidates.sort()
        res = []
        dfs(0, [], 0)
        return res
```

# 131

[Leetcode](https://leetcode.com/problems/palindrome-partitioning/)

### Palindrome Parititioning



```python
class Solution:
    def partition(self, s):
        if not s:
            return []
        res = []
        self.dfs(s, 0, [], res)
        return res
    
    
    
    def dfs(self, s, startIndex, partition, res):
        if len(s) == startIndex:
            res.append([i for i in partition])
            return
        for endIndex in range(startIndex,len(s)):
            substring = s[startIndex:(endIndex + 1)]
            if not self.isPalindrome(substring):
                continue
            partition.append(substring)
            self.dfs(s, endIndex + 1, partition, res)
            partition.pop()
            
    def isPalindrome(self, s):
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
```

# 46

[leetcode](https://leetcode.com/problems/permutations/)

### Permutation




```python
class Solution(object):
    def permute(self, nums):
        def dfs(nums, permutation):
            if not nums:
                res.append(permutation)
                return
            for i in range(len(nums)):
                dfs(nums[:i] + nums[i + 1:], permutation + [nums[i]])
                    
        res = []
        dfs(nums, [])
        return res   
```

# 47

[Leetcode](https://leetcode.com/problems/permutations-ii/)

### Permutations II


```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(nums, sub):
            if len(nums) == 0:
                res.append(sub)
                return
            for i in range(len(nums)):
                if i > 0 and nums[i-1] == nums[i]: continue
                dfs(nums[:i] + nums[i+1:], sub + [nums[i]])
        res = []
        nums.sort()
        dfs(nums, [])
        return res
```

# 17

[Leetcode](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)


### Letter Combinations of a Phone Number



```python
class Solution(object):
    def letterCombinations(self, digits):
        def dfs(digits, ind, cur):
            if len(cur) == len(digits):
                res.append(cur)
                return
            for i in map[digits[ind]]:
                dfs(digits, ind + 1, cur + i)
        if len(digits) == 0: return []
        map = {'2': ['a', 'b', 'c'],
              '3': ['d', 'e', 'f'],
              '4': ['g', 'h', 'i'],
              '5': ['j', 'k', 'l'],
              '6': ['m', 'n', 'o'],
              '7': ['p', 'q', 'r', 's'],
              '8': ['t', 'u', 'v'],
              '9': ['w', 'x', 'y', 'z']}
        res = []
        dfs(digits, 0, '')
        return res
```

# 78

[Leetcode](https://leetcode.com/problems/subsets/)

### Subsets


```python
class Solution:
    def subsets(self, nums):
        res = []
        self.dfs(0, nums, res, [])
        return res   
        
    def dfs(self, ind, nums, res, subset):
        if ind == len(nums):
            res.append([i for i in subset])
            return
        self.dfs(ind + 1, nums, res, subset)
        subset.append(nums[ind])
        self.dfs(ind + 1, nums, res,subset)
        subset.pop()
```

# 90

[Leetcode](https://leetcode.com/problems/subsets-ii/)

### Subset II



```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(ind, sub):
            if ind == len(nums):
                res.append(sub)
                return 
            dfs(ind + 1, sub)
            if ind > 0 and nums[ind] == nums[ind-1] and visited[ind-1] == 0: return
            visited[ind] = 1 
            dfs(ind + 1, sub + [nums[ind]])
            visited[ind] = 0
            
                
        nums.sort()
        res = []
        visited = [0 for _ in range(len(nums))]
        dfs(0, [])
        return res
        
```

# 22

[Leetcode](https://leetcode.com/problems/generate-parentheses/)

### Generate Parentheses



```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        self.dfs(res, 0, 0, '', n)
        return res
        
    def dfs(self, res, l, r, s, max):
        if l + r == max * 2:
            res.append(s)
            return
        
        if l < max:
            self.dfs(res, l + 1, r, s + '(', max)
        if r < l:
            self.dfs(res, l, r + 1, s + ')', max)
```

# 79

[Leetcode](https://leetcode.com/problems/word-search/)

### Word Search



```python
class Solution(object):
    def exist(self, board, word):
        def dfs(i, j, ind, visited):
            if ind == len(word): return True
            if 0 <= i < m and 0 <= j < n and board[i][j] == word[ind] and visited[i][j] == 0:
                visited[i][j] = 1
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if dfs(i + dx, j + dy, ind + 1, visited): return True
                visited[i][j] = 0
                return False
            
        m, n = len(board), len(board[0])
        visited = [[0] * n for _ in range(m)]
        return any([dfs(i, j, 0, visited) for i in range(m) for j in range(n) if board[i][j] == word[0]])
```

# 212

[Leetcode](https://leetcode.com/problems/word-search-ii/)

### Word Search II



```python

```

### Word Ladder II

126
https://leetcode.com/problems/word-ladder-ii/

37

310

# 51
### N-Queens
[Leetcode](https://leetcode.com/problems/n-queens/)



- 拆解问题 用多个function合成
- 如果两个坐标和是一样的 就是斜线


```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def dfs(n, cur):
            if len(cur) == n:
                res.append(drawchess(cur, n))
                return
            for i in range(n):
                if isValid(cur,i): 
                    dfs(n, cur + [i])
        
        def isValid(cur, i):
            if i in cur: return False
            for ind, v in enumerate(cur):
                if abs(ind - len(cur)) == abs(i - v): return False
            return True
        
        def drawchess(cur, n):
            chess = ['.' * n for i in range(n)]
            for ind, v in enumerate(cur):
                chess[ind] = chess[ind][:v] + 'Q' + chess[ind][v+1:]
            return chess
        
        res = []
        dfs(n, [])
        return res
```

# 52
### N-Queens II

[Leetcode](https://leetcode.com/problems/n-queens-ii/)

- 如果两个坐标和是一样的 就是斜线


```python
class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        def isValid(cur, i):
            if i in cur: return False
            for ind, v in enumerate(cur):
                if abs(ind - len(cur)) == abs(i - v): return False
            return True
            
        def dfs(n, cur):
            if len(cur) == n: 
                res[0] += 1
                return 
            for i in range(n):
                if isValid(cur, i): dfs(n, cur+[i])
        
        res = [0]
        dfs(n, [])
        return res[0]
```

### Flatten Binary Tree to Linked List

114.
https://leetcode.com/problems/flatten-binary-tree-to-linked-list/


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root):
        """
        Do not return anything, modify root in-place instead.
        """
        prev = [None]
        
        def dfs(root, prev):
            if not root: return
            left, right = root.left, root.right
            if prev[0]:
                prev[0].left = None
                prev[0].right = root
            prev[0] = root
            dfs(left, prev)
            dfs(right, prev)
        
        dfs(root, prev)
```

# 77

[Leetcode](https://leetcode.com/problems/combinations/)

### Combinations 



```python
class Solution(object):
    def combine(self, n, k):
        def dfs(pos, subset, length):
            if length == k:
                res.append(subset)
                return 
            for i in range(pos, n + 1):
                dfs(i + 1, subset+[i], length + 1)
        res = []
        dfs(1, [], 0)
        return res
```

# 1059

[Leetcode](https://leetcode.com/problems/all-paths-from-source-lead-to-destination/)

### All Paths from Source Lead to Destination


```python
class Solution(object):
    def leadsToDestination(self, n, edges, source, destination):
        """
        :type n: int
        :type edges: List[List[int]]
        :type source: int
        :type destination: int
        :rtype: bool
        """
        
        def dfs(node):
            if visited[node] == 1: return True
            elif visited[node] == -1: return False
            elif len(graph[node]) == 0: return node == destination
            else: 
                visited[node] = -1
                for i in graph[node]:
                    if not dfs(i): return False
                visited[node] = 1
                return True
        
        graph = {i:[] for i in range(n)}
        visited = [0] * n
        for a, b in edges:
            graph[a].append(b)
        return dfs(source)
```

# 1415
### The k-th Lexicographical String of All Happy Strings of Length n
[Leetcode](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/)


```python
class Solution(object):
    def getHappyString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        def dfs(n, cur):
            if len(cur) == n:
                res.append(cur)
                return
            for i in ['a', 'b', 'c']:
                if cur and i == cur[-1]: continue
                dfs(n, cur + i)
                if len(res) == k: return 
                    
        res = []
        dfs(n, '')
        return res[-1] if len(res) == k else ""
```

# 329
### Longest Increasing Path in a Matrix
[Leetcode](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)


```python
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        def dfs(i, j):
            path_num = 0
            if memo[i][j] != 0: return memo[i][j]
            for dx, dy in [(-1, 0),(1, 0),(0, -1),(0, 1)]:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                    path_num = max(path_num, dfs(x, y))
            memo[i][j] = path_num + 1
            return memo[i][j]

        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for i in range(m)] 
        res = 0
        for i in range(m):
            for j in range(n):
                res = max(res, dfs(i, j))
        return res
```

# 124

[Leetcode](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

### Binary Tree Maximum Path Sum


```python

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(root):
            if not root: return 0
            left = right = 0
            if root.left: left = max(0, dfs(root.left))
            if root.right: right = max(0, dfs(root.right))
            res[0] = max(res[0], left + root.val + right)
            return max(left, right) + root.val
                
        res = [-float('inf')]
        dfs(root)
        return res[0]
```

# 494
### Target Sum
[Leetcode](https://leetcode.com/problems/target-sum/)
- dfs会tle
- 最好使用dp
- 加个memo

```python
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        def dfs(nums, sid, cur):
            if sid == len(nums) and cur == S:
                res[0] += 1
                return
            if sid < len(nums):
                dfs(nums, sid + 1, cur + nums[sid])
                dfs(nums, sid + 1, cur - nums[sid])
        
        res = [0]
        dfs(nums, 0, 0)
        return res[0]
```

# 785
### Is Graph Bipartite?
[Leetcode](https://leetcode.com/problems/is-graph-bipartite/)


```python
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        color = {}
        def dfs(pos):
            for i in graph[pos]:
                if i in color:
                    if color[i] == color[pos]: return False
                else:
                    color[i] = 1 - color[pos]
                    if not dfs(i): return False
            return True
        for i in range(len(graph)):
            if i not in color:
                color[i] = 0
                if not dfs(i): return False
        return True
            
                        
```

# 130
### Surrounded Regions
[Leetcode](https://leetcode.com/problems/surrounded-regions/)


```python
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def dfs(i, j):
            if board[i][j] == 'X': return
            for di, dj in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                x, y = i + di, j + dj
                if 0 <= x < m and 0 <= y < n and visited[x][y] == 0:
                    visited[x][y] = 1
                    if board[x][y] == 'O': dfs(x, y)
            
            
        if not board: return board
        m, n = len(board), len(board[0])
        visited = [[0] * n for i in range(m)]
        for i in range(n):
            if board[0][i] == 'O': dfs(0, i)
            if board[m-1][i] == 'O': dfs(m-1, i)
            visited[0][i], visited[m-1][i] = 1, 1
        for j in range(m):
            if board[j][0] == 'O': dfs(j, 0)
            if board[j][n-1] == 'O': dfs(j, n-1)
            visited[j][0], visited[j][n-1] = 1, 1 
        for i in range(m):
            for j in range(n):
                if visited[i][j] == 0 and board[i][j] == 'O':
                    board[i][j] = 'X'
        return board
        
        
                
```

# 139
### Word Break
[Leetcode](https://leetcode.com/problems/word-break/)


```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        def dfs(sid):
            if sid == len(s): return True
            if not memo[sid]: return False
            res = False
            for i in range(sid, len(s)):
                if s[sid:i+1] in wordDict:
                    res = res or dfs(i + 1)
                    memo[sid] = res
            return res
        memo = [True] * len(s)
        return dfs(0)
        
```

# 694
### Number of Distinct Islands
[Leetcode](https://leetcode.com/problems/number-of-distinct-islands/)


```python
class Solution(object):
    def numDistinctIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def dfs(i, j, di):
            if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                grid[i][j] = 0
                shape.append(di)
                dfs(i + 1, j, 1)
                dfs(i - 1, j, 2)
                dfs(i, j + 1, 3)
                dfs(i, j - 1, 4)
                shape.append(0)
        
        if not grid: return 0
        m, n = len(grid), len(grid[0])
        shapes = set()
        for i in range(m):
            for j in range(n):
                shape = []
                dfs(i, j, 0)
                if shape:
                    shapes.add(tuple(shape))
        return len(shapes)
        
```

# 695
### Max Area of Island
[Leetcode](https://leetcode.com/problems/max-area-of-island/)


```python
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def dfs(i, j):
            if 0 <= i < m and  0 <= j < n and grid[i][j] == 1:
                grid[i][j] = 0
                return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)
            return 0
        
        if not grid: return 
        m, n = len(grid), len(grid[0])
        area = [dfs(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        return max(area) if area else 0
        
```

# 1254
# Number of Closed Islands
[Leetcode](https://leetcode.com/problems/number-of-closed-islands/)


```python
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """    
        def dfs(i, j):
            if 0 <= i < m and 0 <= j < n and grid[i][j] == 0:
                grid[i][j] = 1
                dfs(i + 1, j)
                dfs(i - 1, j)
                dfs(i, j + 1)
                dfs(i, j - 1)
                return 1
        
        if not grid: return
        m, n = len(grid), len(grid[0])
        for i in range(m):
            dfs(i, 0)
            dfs(i, n-1)
        for i in range(n):
            dfs(0, i)
            dfs(m-1, i)
        res = 0
        for i in range(1, m-1):
            for j in range(1, n-1):
                if grid[i][j] == 0:
                    res += dfs(i, j)
        return res
```

# 332
### Reconstruct Itinerary
[Leetcode](https://leetcode.com/problems/reconstruct-itinerary/)


```python
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        def dfs(start, path, res, visited):
            if len(res[0]) == len(tickets) + 1: return
            if len(path) == len(tickets) + 1:
                res[0] = [i for i in path]
                return
            for end in G.get(start, []):
                if visited[(start, end)] >= 1:
                    visited[(start, end)] -= 1
                    dfs(end, path + [end], res, visited)
                    visited[(start, end)] += 1

        
        from collections import defaultdict
        G = defaultdict(list)
        visited = {}
        for i, j in tickets:
            G[i].append(j)
            visited[(i, j)] = visited.get((i, j), 0) + 1
        G = {k: sorted(v) for k, v in G.items()}
        res = [[]]
        dfs('JFK', ['JFK'], res, visited)
        return res[0]
        
        
```

# 797
### All Paths From Source to Target
[Leetcode](https://leetcode.com/problems/all-paths-from-source-to-target/)


```python
class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        def dfs(cur, path, res):
            if path[-1] == len(graph) - 1:
                res.append(path)
                return
            for i in graph[cur]:
                dfs(i, path + [i], res)
    
        res = []
        dfs(0, [0], res)
        return res
```

# 934
### Shortest Bridge
[Leetcode](https://leetcode.com/problems/shortest-bridge/)


```python
class Solution(object):
    def shortestBridge(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        def dfs(i,j):
            if i < 0 or i >= m or j < 0 or j >= m or A[i][j] != 1:
                return
            A[i][j] = 2
            q.append((i, j, 0))
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        if not A: return
        m, n = len(A), len(A[0])
        q = []
        flag = False
        for i in range(m):
            for j in range(n):
                if A[i][j] == 1:
                    dfs(i, j)
                    flag = True
                    break
            if flag: break   

        while q:
            x, y, step = q.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                i, j = x + dx, y + dy
                if 0 <= i < m and 0 <= j < n:
                    if A[i][j] == 1: return step
                    if A[i][j] == 0:
                        A[i][j] = 2
                        q.append((i, j, step + 1))
        return -1            
```


```python
lexiorder number
```


      File "<ipython-input-2-8a504b5bd8d9>", line 1
        lexiorder number
                  ^
    SyntaxError: invalid syntax



# 526
### Beautiful Arrangement
[Leetcode](https://leetcode.com/problems/beautiful-arrangement/)
> 只超过5%的人？？？


```python
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        def dfs(pos, sub, res):
            if pos == n + 1:
                res[0] += 1
                return
            for i in range(1, n + 1):
                if i not in set(sub):
                    if i % pos == 0 or pos % i == 0:
                        sub.append(i)
                        dfs(pos + 1, sub, res)
                        sub.pop()
        
        res = [0]
        dfs(1, [], res)
        return res[0]
```

# 863
### All Nodes Distance K in Binary Tree
[Leetcode](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)
> dfs和bfs同时使用


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        
        def dfs(root, prev):
            if not root: return
            if root.left: 
                left = dfs(root.left, root)
                G[root.val].append(left.val)
            if root.right: 
                right = dfs(root.right, root)
                G[root.val].append(right.val)
            G[root.val].append(prev.val)
            return root
   
        from collections import defaultdict
        G = defaultdict(list)
        dfs(root, TreeNode(None))
        q = [target.val]
        visited = set([target.val])
        while q:
            if K == 0: return [i for i in q if i is not None]
            size = len(q)
            for _ in range(size):
                cur = q.pop(0)
                for i in G[cur]:
                    if i not in visited:
                        q.append(i)
                        visited.add(i)
            K -= 1
        return []

```

# 93
### Restore IP Addresses
[Leetcode](https://leetcode.com/problems/restore-ip-addresses/)


```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def dfs(ind, path):
            if ind == len(s) and len(path) == 4:
                res.append('.'.join(path))
            if ind < len(s) and len(path) < 4:
                for eind in range(ind + 1, ind + 4):
                    if eind - ind > 1 and s[ind] == '0': continue
                    if int(s[ind:eind]) < 256:
                        dfs(eind, path + [s[ind:eind]])
                    
        res = []
        dfs(0, [])
        return res
                
```

# 967
### Numbers With Same Consecutive Differences
[Leetcode](https://leetcode.com/problems/numbers-with-same-consecutive-differences/)


```python
class Solution(object):
    def numsSameConsecDiff(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        def dfs(ind, sub):
            if ind == n:
                res.append(''.join([str(i) for i in sub]))
                return
            for i in range(10):
                if ind == 0 and i == 0: continue
                if ind == 0: dfs(ind + 1, sub + [i])
                elif abs(i - sub[-1]) == k: dfs(ind + 1, sub + [i])
        res = []
        dfs(0, [])
        return res
```

# 1239
### Maximum Length of a Concatenated String with Unique Characters
[Leetcode](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/)


```python
class Solution(object):
    def maxLength(self, arr):
        """
        :type arr: List[str]
        :rtype: int
        """
        def dfs(ind, sub):
            if ind == len(arr):
                res[0]= max(res[0], len(sub))
                return
            if ind < len(arr):
                for i in arr[ind]:
                    if i in sub: break
                if i not in sub and len(arr[ind]) == len(set(arr[ind])): dfs(ind + 1, sub + arr[ind])
                dfs(ind + 1, sub)
            
        res = [0]
        dfs(0, '')
        return res[0]
```

# 1219
### Path with Maximum Gold
[Leetcode](https://leetcode.com/probmlems/path-with-maximum-gold/)


```python
class Solution(object):
    def getMaximumGold(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def dfs(i, j):
            if i < 0 or i >=m or j < 0 or j >= n or grid[i][j] == 0: return 0
            cur = grid[i][j]
            grid[i][j] = 0
            res = max(dfs(i+1, j), dfs(i, j+1), dfs(i-1,j), dfs(i, j-1))
            grid[i][j] = cur
            return res + cur
          
        m, n = len(grid), len(grid[0])
        a = [dfs(i,j) for i in range(m) for j in range(n)]
        print(a)
        return max(a)
```

# 784
### Letter Case Permutation
[Leetcode](https://leetcode.com/problems/letter-case-permutation/)


```python
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        def dfs(ind, sub):
            if len(sub) == len(S): 
                res.append(sub)
                return 
            if S[ind].isalpha():
                dfs(ind + 1, sub + S[ind].lower())
                dfs(ind + 1, sub + S[ind].upper())     
            else: dfs(ind + 1, sub + S[ind])
                
        res = []
        dfs(0, '')
        return res
```

# 1291
### Sequential Digits
[Leetcode](https://leetcode.com/problems/sequential-digits/)

其实用bfs更好


```python
class Solution(object):
    def sequentialDigits(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: List[int]
        """
        def dfs(cur):
            if low <= int(cur) <= high:
                res.append(int(cur))
            if len(cur) > len(str(high)): return
            if int(cur[-1]) + 1 >= 10: return
            dfs(cur + str(int(cur[-1]) + 1))
            
        res = []
        for i in range(1, 10):
            dfs(str(i))
        # print(res, sorted(res))
        return sorted(res)
```

# 472
### Concatenated Words
[Leetcode](https://leetcode.com/problems/concatenated-words/)


```python
class Solution(object):
    def findAllConcatenatedWordsInADict(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        def dfs(word):
            if word in memo: return True
            flag = False
            for i in range(1, len(word)):
                pre = word[:i]
                post = word[i:]
                if (pre in s and post in s) or ((pre in s) and dfs(post)) or ((post in s) and dfs(pre)):
                    memo.add(word)
                    flag = True
                    break
            return flag
        
        s = set(words)
        res = []
        memo = set()
        for word in words:
            if dfs(word): res.append(word)
        return res
```

## [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        def dfs(cur):
            if sum(cur) == n and len(cur) == k:
                res.append([i for i in cur])
                return
            if sum(cur) < n and len(cur) < k:
                for i in range(cur[-1] + 1, 10):
                    dfs(cur + [i])
                

        res = []
        for i in range(1, 10):
            dfs([i])
        return res
```
