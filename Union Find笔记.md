###  总结

- 点的连边
- 有删边操作就不能用union find
- 查询两个元素是否在同一个集合
- 合并两个元素所在的结合
- 判断集合个数
- 判断当前集合中所有元素的个数



### 题型分类

- [547](#547)
    - [Friend Circles](#friend-circles)
- [721](#721)
    - [Accounts Merge](#accounts-merge)
- [990](#990)
    - [Satisfiability of Equality Equations](#satisfiability-of-equality-equations)
- [1061](#1061)
    - [Lexicographically Smallest Equivalent String](#lexicographically-smallest-equivalent-string)
- [1101](#1101)
    - [The Earliest Moment When Everyone Become Friends](#the-earliest-moment-when-everyone-become-friends)
- [737](#737)
    - [Sentence Similarity II](#sentence-similarity-ii)
- [323](#323)
    - [Number of Connected Components in an Undirected Graph](#number-of-connected-components-in-an-undirected-graph)
- [1168](#1168)
    - [Optimize Water Distribution in a Village](#optimize-water-distribution-in-a-village)
- [547](#547-1)
    - [Number of Provinces](#number-of-provinces)
- [323](#323-1)
    - [Number of Connected Components in an Undirected Graph](#number-of-connected-components-in-an-undirected-graph-1)
- [323](#323-2)
    - [Regions Cut By Slashes](#regions-cut-by-slashes)
最小生成树
- [Optimize Water Distribution in a Village](#1168)

### 易错点


```python

```

### 必备模版技巧


```python
def find(self, parent, i):
    if parent[i] == i:
        return i
    parent[i] = self.find(parent[i])
    return parent[i]

def union(self, parent, x, y):
    root_a = self.find(parent, a)
    root_b = self.find(parent, b)
    parent[root_a] = root_b
    return   
```


```python
class UnionFind(object):
    def __init__(self):
        self.parents = {}
    
    def make_set(self, x):
        self.parents[x] = x
        
    def find(self, x):
        if parent[i] == i:
            return i
        parent[i] = self.find(parent[i])
        return parent[i]
            return self.parents[x]
    
    def union(self, x, y):
        root_a = self.find(parent, a)
        root_b = self.find(parent, b)
        parent[root_a] = root_b
        return   
```

---

### Connecting Graph

### Connceting Graph II

### Number of Islands

200
https://leetcode.com/problems/number-of-islands/


```python
class Solution:
    def numIslands(self, grid):
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
        self.count = 0
        parent = [-2 for i in range(m * n)]
        dr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    parent[i * n + j] = -1
                    self.count += 1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    for dx, dy in dr:
                        x, y = i + dx, j + dy
                        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
                            self.union(parent, x * n + y, i * n + j)
        
        return self.count
    
        
    def find(self, parent, i):
        if parent[i] != -1:
            return self.find(parent, parent[i])
        else:
            return i
        
    def union(self, parent, a, b):
        root_a = self.find(parent, a)
        root_b = self.find(parent, b)
        if root_a != root_b:
            parent[root_a] = root_b
            self.count -= 1
        
```

### Number of Islands II


- 注意有重复值的时候该如何操作

305
https://leetcode.com/problems/number-of-islands-ii/submissions/


```python
class Solution:
    def numIslands2(self, m, n):
        
        def find(i, parent):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i], parent)
            return parent[i]

        def union(a, b, parent):
            global count
            root_a, root_b = find(a, parent), find(b, parent)
            if root_a != root_b: 
                parent[root_b] = root_a
                return 1 
            return 0
            
        dr = ((1, 0), (-1, 0), (0, 1), (0, -1))
        parent = [-1 for i in range(m * n)]
        count = 0
        res = []
        for i, j in positions:
            if parent[i * n + j] == -1:
                parent[i * n + j] = i * n + j
                count += 1
                for dx, dy in dr:
                    x, y = i + dx, j + dy
                    if 0 <= x < m and 0 <= y < n and parent[x * n + y] != -1:
                        count -= union(i * n + j, x * n + y, parent)
            print(count)
                    
            res.append(count)
        
        return res
```

# 547

[Leetcode](https://leetcode.com/problems/friend-circles/)

### Friend Circles

- 注意怎么去生成parent， 初始化的时候永远都是自己指向自己
- 合并的时候count应该怎么做， python 里面self。count要用的熟练




```python
class Solution:
    def findCircleNum(self, M):
        n = len(M)
        parent = [i for i in range(n)]
        self.count = len(parent)
        for student, friend_list in enumerate(M):
            for friend in range(len(friend_list)):
                if M[student][friend] == 1:
                    self.union(parent, student, friend)
                    print(self.count)    
        return self.count        
        
    def find(self, parent, i):
        if parent[i] == i:
            return i
        parent[i] = self.find(parent, parent[i])
        return parent[i]
    
    def union(self, parent, a, b):
        root_a = self.find(parent, a)
        root_b = self.find(parent, b)
        if root_a != root_b: 
            parent[root_b] = root_a
            self.count -= 1
        return
```

# 721
(Leetcode)(https://leetcode.com/problems/accounts-merge/)

### Accounts Merge


```python
class Solution(object):
    def accountsMerge(self, accounts):
        """
        :type accounts: List[List[str]]
        :rtype: List[List[str]]
        """
        def find(parent, i):
            if parent[i] == i:
                return i
            parent[i] = find(parent, parent[i])
            return parent[i]
        def union(parent, a, b):
            root_a, root_b = find(parent, a), find(parent, b)
            if root_a != root_b: parent[root_b] = root_a
            return
        
        id_name = {}
        email_id = {}
        parent = []
        id = 0
        for account in accounts:
            name = account[0]
            for email in account[1:]:
                if email not in email_id.keys():
                    email_id[email] = id
                    id_name[id] = name
                    parent.append(id)    
                    id  +=  1
                union(parent, email_id[account[1]], email_id[email])
            
        res  =  {}
        for email, id in email_id.items():
            master = find(parent, id)
            res[master] =  res.get(master, [id_name[master]]) + [email]
        return [[value[0]] + sorted(value[1:])  for key, value in res.items()]
            
```

# 990

[Leetcode](https://leetcode.com/problems/satisfiability-of-equality-equations/)

### Satisfiability of Equality Equations


```python
class Solution(object):
    def equationsPossible(self, equations):
        """
        :type equations: List[str]
        :rtype: bool
        """
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(a, b):
            root_a, root_b =  find(a), find(b)
            if root_a != root_b: parent[root_b] = root_a
            return
        
        parent = {a: a for a in string.lowercase}
        for x, s, _, y in equations:
            if s == '=': union(x, y)
        for x, s, _, y in equations:
            if s == '!' and find(x) == find(y): 
                return False
        return True
```

# 1061

[Leetcode](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/)

### Lexicographically Smallest Equivalent String


```python
class Solution(object):
    def smallestEquivalentString(self, A, B, S):
        """
        :type A: str
        :type B: str
        :type S: str
        :rtype: str
        """
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a != root_b: 
                if root_a < root_b: parent[root_b] = root_a
                else: parent[root_a] = root_b
            return
        
        parent = {a: a for a in string.lowercase}
        for i, j in zip(A,B):
            union(i,j)
        return ''.join([find(i) for i in S])
```

# 1101

[Leetcode](https://leetcode.com/problems/the-earliest-moment-when-everyone-become-friends/)
### The Earliest Moment When Everyone Become Friends


```python
class Solution(object):
    def earliestAcq(self, logs, N):
        """
        :type logs: List[List[int]]
        :type N: int
        :rtype: int
        """

        def find(i):
            if i == parent[i]: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(a, b, count):
            root_a, root_b = find(a), find(b)
            if root_a != root_b: 
                parent[root_b] = root_a
                count -= 1
            return count
        
        parent = [i for i in range(N)]
        count = N
        logs = sorted(logs, key = lambda x: x[0])
        for timestamp, a, b in logs:
            count = union(a,b, count)
            if count == 1: return timestamp
        return -1
```

# 737 

[Leetcode](https://leetcode.com/problems/sentence-similarity-ii/)

### Sentence Similarity II


```python
class Solution(object):
    def areSentencesSimilarTwo(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                if root_a < root_b: parent[root_b] = root_a
                else:  parent[root_a] = root_b
            return 
        
        if len(words1) != len(words2): return False
        parent = {}
        for i, j in pairs:
            if i not in parent.keys(): parent[i] =  i
            if j not in parent.keys(): parent[j] =  j
            union(i, j)
        for i, j in zip(words1, words2):
            if i == j: continue
            if i not in parent.keys() or j not in parent.keys() or find(i) != find(j):
                return False
        return True
```

# 323

[Leetcode](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

### Number of Connected Components in an Undirected Graph


```python
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        def find(i):
            if i == parent[i]: return i
            parent[i] = find(parent[i])
            return parent[i]

        count = n
        parent = [i for i in range(n)]
        for i, j in edges:
            x, y = find(i), find(j)
            if x!= y: 
                parent[y] = x
                count -= 1
        return count
```

# 1168
[Leetcode](https://leetcode.com/problems/optimize-water-distribution-in-a-village/)

### Optimize Water Distribution in a Village


```python
class Solution(object):
    def minCostToSupplyWater(self, n, wells, pipes):
        """
        :type n: int
        :type wells: List[int]
        :type pipes: List[List[int]]
        :rtype: int
        """
        parent = {i: i for i in range(n + 1)}
        def find(x):
            if x == parent[x]: return x
            parent[x] = find(parent[x])
            return parent[x]
        
        well = [(c, 0, h) for h, c in enumerate(wells, 1)]
        pipe = [(c, a, b) for a, b, c in pipes]
        res = 0
        for c, x, y in sorted(well + pipe):
            x, y = find(x), find(y)
            if x != y:
                parent[find(x)] = find(y)
                res += c
                n -= 1
            if n == 0:
                return res
                
        
```

# 547
### Number of Provinces
[Leetcode](https://leetcode.com/problems/number-of-provinces/)


```python
class Solution(object):
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b
            return root_b
        a
        def find(a):
            if parent[a] == a: return a
            parent[a] = find(parent[a])
            return parent[a]

        if not M: return
        count = n = len(M)
        parent = [i for i in range(n)]
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1: union(i, j)
        return sum([1 if i == v else 0 for i, v in enumerate(parent) ])
                    
```

# 323
### Number of Connected Components in an Undirected Graph
[Leetcode](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)


```python
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        def find(a):
            if parent[a] == a: return a
            parent[a] = find(parent[a])
            return parent[a]
        
        def union(a, b):
            root_a, root_b = find(a), find(b)
            parent[root_b] = root_a
            if root_a != root_b: self.count -= 1
        
        self.count = n
        parent = [i for i in range(n)]
        for a, b in edges:
            union(a, b)
        return  self.count
```

# 323 
### [Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)
- 拆分成4个小的cell
```python
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a == root_b: return 
            parent[root_b] = root_a
            cnt[0] -= 1
            
        parent = {}
        cnt = [0]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                for k in range(4):
                    parent[(i, j, k)] = (i, j, k)
                    cnt[0] += 1
                if i: union((i, j, 0), (i-1, j, 2))
                if j: union((i, j, 3), (i, j - 1, 1))
                if grid[i][j] == '/':
                    union((i, j, 0), (i, j, 3))
                    union((i, j, 1), (i, j, 2))
                elif grid[i][j] == '\\':
                    union((i, j, 3), (i, j, 2))
                    union((i, j, 0), (i, j, 1))
                if grid[i][j] == ' ': 
                    union((i, j, 1), (i, j, 2))
                    union((i, j, 2), (i, j, 3))
                    union((i, j, 3), (i, j, 0))
        return cnt[0]
```