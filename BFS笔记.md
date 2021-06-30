### 总结

- 什么时候用bfs？
    - 图的遍历 
        - 层级遍历 level order
        - 由点及面 connected component
        - 拓扑排序 topological sorting
    - 最短路径
        - 仅限简单图求最短路径
        - 每条边都是1的长度， 且没有方向
        
- 是否需要层级遍历
    - size = queue.size()    
        
- bfs在二叉树
    - binary tree serialization
    
- bfs在图
    -  n-1条边 n个点连通


- bfs在矩阵
    - delta x， delta y
        - {0, 1, -1, 0}
        - {1, 0, 0, -1}
    - inbound
    
假设一个图有N个节点和M条边, BFS会走遍所有节点, 时间是O(N), 然后由于每个节点会检查所有的出边, 最终所有的边都会被检查过, 时间是O(M), 所以BFS的时间复杂度是O(N+M).

队列里面最多可能存放所有节点, 空间复杂度为O(N).

### 题型分类

- #### 需要引入depth

    - [Word Ladder](#127)
    - [Snakes and Ladders](#909)
    
    


- ####  引入size
    
    - [N-ary Tree Level Order Traversal](#429)
    
    
    
    
- #### 由点及面
    - 从0出发更省力
    - [01Matrix](#542)
    - [Walls and Gates](#286)
    - 01结构 可以不设置visited变量，直接在input的matrix上改
    - [Number of Islands](#200)(可dfs)
    - [Shortest Distance from All Buildings](#317)
    - [Rotten Oranges](#994)
    - [The maze](#490)
    - [Shortest Path in Binary Matrix](#1091)
    - [As Far from Land as Possible](#1162)
    - 了解如何剪枝
    - [Minimum knight moves](#1197)
    - [Pacific Atlantic Water Flow](#417)
    - 其他
    - [Minimum Moves to Reach Target with Rotations](#1210)
    - [Sliding Puzzle](#773)

    
- #### graph
    - 构建图， 同时需要考虑一下degree
    - [Employee Importance](#690)
    - [Graph Valid Tree](#261)
    - [Minimum Height Tree](#310)
    - [Bus Routes](#815)
    
    
- #### Topological Sort
    - [Course Schedule](#207)
    - [Course Schedule II](#210)
    - [Sequence Reconstruction](#444)
    - [Alien Dictionary](#269)

- #### 也可dfs
    - [Clone Graph](#133)

### 易错点

### 必备模版技巧

---

# 127

[Leetcode](https://leetcode.com/problems/word-ladder/)


### Word Ladder


- 隐式图搜索
- 简单图搜索路径
- Time Complexity O(n*26^wordLength)
- Space Complexity O(n)
> 对wordlist转换成set，使得搜索时间变成o（1）


```python
# - TLE
class Solution:
    def ladderLength(self, beginWord,endWord):
        wordList = set(wordList)
        queue = [(beginWord, 1)]
        alpha = string.ascii_lowercase
        visited = set()
        while queue:
            cur, depth = queue.pop(0)
            if cur == endWord: return depth
            for i in range(len(cur)):
                for ch in alpha:
                    new_word =  cur[:i] + ch + cur[i + 1:]
                    if new_word in wordList and new_word not in visited:
                        queue.append((new_word, depth + 1))
                        visited.add(new_word)
        return 0
```

# 261

[Leetcode](https://leetcode.com/problems/graph-valid-tree/)

### Graph valid tree

- how to build a tree
- check if $edges = n - 1
- check if all the nodes are visited






```python
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        from collections import deque
        if len(edges) != n - 1:
            return False
        
        graph = self.buildgraph(n, edges)
        queue = deque([0])
        visited = set([0])
        # bfs
        while queue:
            point = queue.popleft()
            for neighbor in graph[point]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == n


    def buildgraph(self, n, edges):
        dict = {}
        for i in range(n):
            dict[i] = []
        
        for edge in edges:
            dict[edge[0]].append(edge[1])
            dict[edge[1]].append(edge[0])  
        
        return dict
```

# 133
### Clone graph

[Leetcode](https://leetcode.com/problems/clone-graph/)
- 用一个map 一一对应把它存起来
- 相似做法还有一个linked list的题


```python
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        from collections import deque
        if not node:
            return node
        visited = {}
        visited[node] = Node(node.val,[])
        queue = deque([node])
        
        while queue:
            cur = queue.popleft()
            for neighbor in cur.neighbors:
                if neighbor not in visited.keys():
                    visited[neighbor] = Node(neighbor.val,[])
                    queue.append(neighbor)
                visited[cur].neighbors.append(visited[neighbor])
                
        return visited[node]
```

### Search Graph Nodes


```python
class solution(object):
    def search(self, node, target):
        from collections import deque
        if not node:
            return node
        queue = deque(node)
        visited = set(node)
        while queue:
            cur = queue.popleft()
            for neighbor in cur.neighbors:
                if neighbor not in visited:
                    if neighbor.val == target:
                        return neighbor
                    queue.append(neighbor)
                    
        return False
```

### Topological Sorting

- 又循环依赖就不能拓扑排序


```python
class solution(object):
    
```

### zombie in matrix


```python

```

# 200


[Leetcode](https://leetcode.com/problems/number-of-islands/)

### Number of Islands

- 也可dfs，uninon find


```python
# bfs
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
    
        def bfs(i, j):
            q, grid[i][j] = [(i,j)], '0'
            for i, j in q:
                for x, y in [(i - 1, j),(i + 1, j),(i, j - 1),(i, j + 1)]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
                        grid[x][y] = '0'
                        q.append((x,y))
            return 1
        
        return sum(bfs(i,j) for i in range(m) for j in range(n) if grid[i][j] == '1')
```


```python
# dfs
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return
            grid[i][j] = '0'
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
            return 1

        return sum(dfs(i, j) for i in range(m) for j in range(n) if grid[i][j] == '1')   
```


```python
# union find
```

 ### Knight shortest path

### build post office

# 994

[Leetcode](https://leetcode.com/problems/rotting-oranges/)

### Rotting Orange


```python
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid) == 0: return 0
        m, n = len(grid), len(grid[0])
        q = []
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh += 1
        time = 0
        while q:
            x, y, time = q.pop(0)
            for new_x, new_y in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] == 1:
                    grid[new_x][new_y] = 2
                    fresh -= 1
                    q.append((new_x, new_y, time + 1))
        return time if fresh == 0 else -1
```

# 690

[Leetcode](https://leetcode.com/problems/employee-importance/)

### Employee Importance



```python
# bfs
class Solution:
    def getImportance(self, employees):
        employee_dict = {}
        importance_dict = {}
        for employee in employees:
            employee_dict[employee.id] = employee.subordinates
            importance_dict[employee.id] = employee.importance
            
        importance = importance_dict[id]
        queue = employee_dict[id]
        while queue:
            subordinate = queue.pop(0)
            importance += importance_dict[subordinate]
            queue.extend(employee_dict[subordinate])
            
        return importance
```


```python
# dfs
class Solution:
    def getImportance(self, employees):
        employee_dict = {}
        importance_dict = {}
        for employee in employees:
            employee_dict[employee.id] = employee.subordinates
            importance_dict[employee.id] = employee.importance
        
        importance = [importance_dict[id]]
        self.helper(employee_dict[id], employee_dict, importance_dict, importance)
        return importance[0]
            

    def helper(self,subordinates, employee_dict, importance_dict, importance):
        if not subordinates:
            return
        for i in subordinates:
            print(i)
            importance[0] += importance_dict[i]
            self.helper(employee_dict[i], employee_dict, importance_dict, importance)
```

# 1162

### As Far from Land as Possible

[Leetcode](https://leetcode.com/problems/as-far-from-land-as-possible/)


```python
class Solution:
    def maxDistance(self, grid):
        n = len(grid)
        queue = []
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))                
        count = -1
        while queue and len(queue) != n ** 2:
            level = []
            for x, y in queue:
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < n and 0 <= j < n and grid[i][j] == 0:
                        grid[i][j] = 1
                        level.append((i,j))
            queue = level
            count += 1
    
        return count
                    
```

# 1161
### Maximum Level Sum of a Binary Tree    

[Leetcode](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/)


```python
class Solution:
    def maxLevelSum(self, root):
        if not root: return 0
        
        queue = [root]
        level = 0
        max_sum = -float('inf')
        max_level = 0
        while queue:
            size = len(queue)
            level += 1
            sum = 0
            for i in range(size):
                node = queue.pop(0)
                sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if sum > max_sum: 
                max_level = level
                max_sum = sum
        
        return max_level
        
```

# 542
### 01 Matrix
[Leetcode](https://leetcode.com/problems/01-matrix/)



```python
class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if not matrix: return matrix
        m, n = len(matrix), len(matrix[0])
        q = []
        visited = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    q.append((i, j, 0))
                    visited[i][j] = 1
        
        while q:
            i, j, step = q.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and visited[x][y] == 0:
                    matrix[x][y] = step + 1
                    q.append((x, y, step + 1))
                    visited[x][y] = 1
        return matrix
```

# 841

### Keys and Rooms
[Leetcode](https://leetcode.com/problems/keys-and-rooms/)


```python
class Solution:
    def canVisitAllRooms(self, rooms):
        visited = [0 for i in range(len(rooms))]
        queue = [rooms[0]]
        visited[0] = 1
        
        
        while queue:
            cur = queue.pop(0)
            if 0 not in set(visited): return True
            for i in cur:
                if visited[i] == 0:
                    queue.append(rooms[i])
                    visited[i] = 1
                    
        return False
```

# 1091

### Shortest Path in Binary Matrix
[Leetcode](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
> 注意corner case：当起点是1该怎么做


```python
class Solution:
    def shortestPathBinaryMatrix(self, grid):
        n = len(grid)
        if grid[0][0] == 1: return -1
        queue = [((0,0), 0)]
        dr = [(1, 1),(1, 0),(1, -1),(0, 1),(0, -1),(-1, 1),(-1, 0),(-1, -1)]
        while queue:
            cur, depth = queue.pop(0)
            print(cur, depth)
            if cur[0] == n - 1 and cur[1] == n - 1: return depth + 1
            for dx, dy in dr:
                new_x, new_y = cur[0] + dx, cur[1] + dy
                if 0 <= new_x < n and 0 <= new_y < n and grid[new_x][new_y] == 0:
                    queue.append(((new_x, new_y), depth + 1))
                    grid[new_x][new_y] = 1
                    
                    
        return -1
```

# 752
### Open the lock
https://leetcode.com/problems/open-the-lock/
> Time: O(10**4) 

> Space: O(10**4)


```python
class Solution:
    def openLock(self, deadends):
        deadends_set = set(deadends)
        if '0000' in deadends_set: return -1
        queue = [('0000', 0)]
        visited = set('0000')
        while queue:
            cur, depth = queue.pop(0)
            if cur == target: return depth
            for i in range(len(cur)):
                for j in [1,-1]:
                    new_num = (int(cur[i]) + j + 10) % 10
                    new_cur = cur[:i] + str(new_num) + cur[i + 1:]
                    if new_cur not in deadends_set and new_cur not in visited:
                        queue.append((new_cur, depth + 1))
                        visited.add(new_cur)

                        
        return -1
```

# 815
### Bus Routes
[Leetcode](https://leetcode.com/problems/bus-routes/)



```python
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        """
        :type routes: List[List[int]]
        :type S: int
        :type T: int
        :rtype: int
        """
        if S == T: return 0
        from collections import defaultdict
        G = defaultdict(set)
        for i in range(len(routes)):
            for j in range(i,len(routes)):
                if len(set(routes[i]) & set(routes[j])) > 0:
                    G[i].add(j)
                    G[j].add(i)
        q = [(ind, 1) for ind, route in enumerate(routes) if S in route ]
        visited = set()
        while q:
            cur, step = q.pop(0)
            if T in routes[cur]: return step
            for i in G[cur]:
                if i not in visited:
                    visited.add(i)
                    q.append((i, step + 1))
        return -1
```

# 773
### Sliding Puzzle

[Leetcode](https://leetcode.com/problems/sliding-puzzle/)
- queue里面装的是一整个board， 用serialize的方式进行判断是否已经visited


```python
class Solution:
    def slidingPuzzle(self, board):
        m = len(board)
        n = len(board[0])
        queue = [(board, 0)]
        visited = set()
        target = '123450'
        while queue:
            cur, depth = queue.pop(0)
            if self.serialize(cur) == target: return depth
            for i in range(m):
                for j in range(n):
                    if cur[i][j] == 0:
                        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            next_cur = [row[:] for row in cur]
                            new_i, new_j = i + dx, j + dy
                            if 0 <= (i + dx) < m and 0 <= (j + dy) < n:
                                next_cur[i][j], next_cur[new_i][new_j] = next_cur[new_i][new_j], next_cur[i][j]
                                if self.serialize(next_cur) not in visited:
                                    queue.append((next_cur, depth + 1))
                                    visited.add(self.serialize(next_cur))
                                    
        return -1
        
        
    def serialize(self, board):
        serial = ''
        for row in board:
            for num in row:
                serial += str(num)
        return serial        
```

# 934
### Shortest Bridge

[Leetcode](https://leetcode.com/problems/shortest-bridge/)


```python
class Solution:
    def shortestBridge(self, A):
        
        def dfs(queue, A, i, j):
            A[i][j] = 0
            visited[i][j] = 1
            for dx, dy in dr:
                new_x, new_y = i + dx, j + dy
                if 0 <= new_x < m and 0 <= new_y < n and A[new_x][new_y] == 1:
                    queue.append(((new_x, new_y), -1))
                    dfs(queue, A, new_x, new_y)  
        
        
        m = len(A)
        n = len(A[0])
        queue = []
        visited = [[0 for i in range(n)] for i in range(m)]
        dr =  ((1, 0), (-1 ,0), (0, 1), (0, -1))
        flag = False
        for i in range(m):
            for j in range(n):
                if A[i][j] == 1:
                    queue.append(((i,j), -1))
                    dfs(queue, A, i, j)
                    flag = True
                    break
            if flag:
                break
          
        print(queue)
        while queue: 
            cur, depth = queue.pop(0) 
            if A[cur[0]][cur[1]] == 1: return depth
            for dx, dy in dr:
                new_x,  new_y = cur[0] + dx,  cur[1] + dy
                if 0 <= new_x < m and 0 <= new_y < n and visited[new_x][new_y] == 0:
                    queue.append(((new_x, new_y), depth + 1))
                    visited[new_x][new_y] = 1

        return -1
```

# 207

[Leetcode](https://leetcode.com/problems/course-schedule/)

### Course Schedule


```python
class Solution:
    def canFinish(self, numCourses, prerequisites):
        G = [[] for i in range(numCourses)]
        indegree = [0] * numCourses
        for j, i in prerequisites:
            G[i].append(j)
            indegree[j] += 1
        queue = [i for i in range(numCourses) if indegree[i] == 0]
        
        order = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for j in G[cur]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)
                        
        return len(order) == numCourses
```

# 210

[Leetcode](https://leetcode.com/problems/course-schedule-ii/)

### Course Schedule II




```python
class Solution:
    def findOrder(self, numCourses, prerequisites):
        G = [[] for i in range(numCourses)]
        indegree = [0] * numCourses
        for j, i in prerequisites:
            G[i].append(j)
            indegree[j] += 1
        
        queue = [i for i in range(numCourses) if indegree[i] == 0]
        order = []
        
        
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for i in G[cur]:
                indegree[i] -= 1
                if indegree[i] == 0: 
                    queue.append(i)
                    
        return order if len(order) == numCourses else []
```

# 269
### Alien Dictionary

[Leetcode](https://leetcode.com/problems/alien-dictionary/)


```python
class Solution(object):
    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        from collections import defaultdict
        G =  defaultdict(list)
        indegree = {}
        for word in words:
            for ch in word:
                indegree[ch] = indegree.get(ch, 0)
        for w1, w2 in zip(words[:-1], words[1:]):
            for a, b in zip(w1, w2):
                if a != b:
                    G[a].append(b)
                    indegree[b] = indegree.get(b, 0) + 1
                    break
                
        # print (G)
        # print(indegree)
        q = [i for i in indegree.keys() if indegree[i] == 0]
        order = []
        while q:
            cur = q.pop(0)
            order.append(cur)
            for i in G[cur]:
                indegree[i] -= 1
                if indegree[i] == 0:
                    q.append(i)
        # print(order)
        return ''.join(order) if len(order) == len(indegree) else ""
            
```

### Longest Increasing Path in a Matrix

329
https://leetcode.com/problems/longest-increasing-path-in-a-matrix/

- 先化成图
- 计算最长距离：中间不用推出，知道queue里面所有东西都跑出 就结束循环


```python
class Solution:
    def longestIncreasingPath(self, matrix) -> int:
        m = len(matrix)
        if m == 0: return 0
        n = len(matrix[0])
        G = collections.defaultdict(set)
        indegree = collections.defaultdict(set)
        num_set = set()
        dr = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for i in range(m):
            for j in range(n):
                num_set.add((i, j))
                for dx, dy in dr:
                    x, y = dx + i, dy + j
                    if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                        G[(i, j)].add((x, y))
                        indegree[(x, y)].add((i, j))
        indegree = {key: len(value) for key, value in indegree.items()}
        queue = [(i ,j) for i, j in num_set if (i, j) not in indegree.keys()]
        max_path = 0
        while queue:
            max_path += 1
            for _ in range(len(queue)):
                node = queue.pop(0)
                for neigh in G[node]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        queue.append(neigh)
                        
        return max_path  
```

# 317

[Leetcode](https://leetcode.com/problems/shortest-distance-from-all-buildings/)

### Shortest Distance from All Buildings


```python
class Solution(object):
    def shortestDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        if not grid or not grid[0]: return -1
        M, N, buildings = len(grid), len(grid[0]), sum(val for line in grid for val in line if val == 1)
        hit, distSum = [[0] * N for i in range(M)], [[0] * N for i in range(M)]

        def BFS(start_x, start_y):
            visited = [[False] * N for k in range(M)]
            visited[start_x][start_y], count1, queue = True, 1, collections.deque([(start_x, start_y, 0)])
            while queue:
                x, y, dist = queue.popleft()
                for i, j in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= i < M and 0 <= j < N and not visited[i][j]:
                        visited[i][j] = True
                        if not grid[i][j]:
                            queue.append((i, j, dist + 1))
                            hit[i][j] += 1
                            distSum[i][j] += dist + 1
                        elif grid[i][j] == 1:
                            count1 += 1
            return count1 == buildings  

        for x in range(M):
            for y in range(N):
                if grid[x][y] == 1:
                    if not BFS(x, y): return -1
        return min([distSum[i][j] for i in range(M) for j in range(N) if not grid[i][j] and hit[i][j] == buildings] or [-1])

```

# 490

[Leetcode](https://leetcode.com/problems/the-maze/)

### The Maze


```python
class Solution(object):
    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        if len(maze) == 0: return False
        m, n = len(maze), len(maze[0])
        q = [(start[0], start[1])]
        visited = set((start[0], start[1]))
        while q:
            x, y = q.pop(0)
            if [x, y] == destination: return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                while 0 <= new_x < m and 0 <= new_y < n and maze[new_x][new_y] != 1:
                    new_x += dx
                    new_y += dy
                new_x -= dx
                new_y -= dy
                if maze[new_x][new_y] == 0 and (new_x, new_y) not in visited:
                    q.append((new_x, new_y))
                    visited.add((new_x, new_y))
        return False
```

# 444

[Leetcode](https://leetcode.com/problems/sequence-reconstruction/)

### Sequence Reconstruction


```python
class Solution(object):
    def sequenceReconstruction(self, org, seqs):
        """
        :type org: List[int]
        :type seqs: List[List[int]]
        :rtype: bool
        """
        values = {x for seq in seqs for x in seq}
        graph = {x: [] for x in values}
        indegree = {x: 0 for x in values}
        for seq in seqs:
            for i in range(len(seq) - 1):
                s = seq[i]
                t = seq[i+1]
                graph[s].append(t)
                indegree[t] += 1
        q = []
        for k, v in indegree.items():
            if v == 0: q.append(k)
        res = []

        while q:
            if len(q) != 1: return False
            cur = q.pop()
            res.append(cur)
            for i in graph[cur]:
                indegree[i] -= 1
                if indegree[i] == 0:
                    q.append(i)
        
        return res == org and len(org) == len(graph.keys()) 
                    
            
        
```

# 1197
### Minimum Knight Moves
[Leetcode](https://leetcode.com/problems/minimum-knight-moves/)
有个剪枝的操作


```python
class Solution(object):
    def minKnightMoves(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        dir = [(-1, -2),(-2, -1),(-1, 2),(-2, 1),(1, 2),(1, -2),(2, 1),(2, -1)]
        q = [(0, 0, 0)]
        visited = {(0,0)}
        while q:
            i, j, time = q.pop(0)
            if i == abs(x) and j == abs(y): return time
            for dx, dy in dir:
                new_i, new_j = i + dx, j + dy
                if (new_i, new_j) not in visited and new_i>-4 and new_j > -4:
                    q.append((new_i,new_j,time+1))
                    visited.add((new_i, new_j))
        
```

# 286
### Walls and Gates
[Leetcode](https://leetcode.com/problems/walls-and-gates/)
> Time: O(mn)

> Space: O(mn)


```python
class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        """
        if not rooms: return
        m, n = len(rooms), len(rooms[0])
        q = []
        visited = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    q.append((i, j, 0))
        while q:
            i, j, step = q.pop(0)
            for dx, dy in [(1, 0),(0, 1),(-1, 0),(0, -1)]:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and visited[x][y] == 0 and rooms[x][y] == 2**31 - 1:
                    rooms[x][y] = step + 1
                    visited[x][y] = 1
                    q.append((x, y, step + 1))
        return rooms
            
        
```

# 417
### Pacific Atlantic Water Flow
[Leetcode](https://leetcode.com/problems/pacific-atlantic-water-flow/)


```python
class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if not matrix: return
        m, n = len(matrix), len(matrix[0])
        q = []
        Avisited = [[0] * n for _ in range(m)]
        Pvisited = [[0] * n for _ in range(m)]
        for i in range(m):
            q.append((i,n-1,'A'))
            q.append((i,0, 'P'))
            Pvisited[i][0] = 1
            Avisited[i][n-1] = 1
        for i in range(n):
            q.append((0, i, 'P'))
            q.append((m-1, i, 'A'))
            Pvisited[0][i] = 1
            Avisited[m-1][i] = 1
        while q:
            i, j, src = q.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = i + dx, j +dy
                if 0 <= x < m and 0 <= y < n and matrix[x][y] >= matrix[i][j]:
                    if src == 'P' and Pvisited[x][y] == 0: 
                        Pvisited[x][y] = 1
                        q.append((x, y,'P'))
                    if src == 'A' and Avisited[x][y] == 0: 
                        Avisited[x][y] = 1
                        q.append((x, y,'A'))
        res = []
        for i in range(m):
            for j in range(n):
                if Pvisited[i][j] == 1 and Avisited[i][j] == 1:
                    res.append([i, j])
        return res
```

# 733
### Flood Fill
[Leetcode](https://leetcode.com/problems/flood-fill/)


```python
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        if not image: return
        m, n = len(image), len(image[0])
        oldColor =  image[sr][sc]
        image[sr][sc] = newColor
        q = [(sr, sc)]
        while q:
            x, y = q.pop(0)
            for dx, dy in [(1, 0),(-1, 0),(0, 1),(0, -1)]:
                i, j = x + dx, y + dy
                if 0 <= i < m and 0 <= j < n and image[i][j] == oldColor and image[i][j] != newColor:
                    image[i][j] = newColor
                    q.append((i,j))
        return image
```

# 310
### Minimum Height Trees
[Leetcode](https://leetcode.com/problems/minimum-height-trees/)


```python
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        from collections import defaultdict
        if not edges: return [0]
        G = defaultdict(set)
        degree = [0] * n
        for i, j in edges:
            G[i].add(j)
            G[j].add(i)
            degree[i] += 1
            degree[j] += 1
        q = [ind for ind, v in enumerate(degree) if v == 1]
        while q:
            size, level = len(q), []
            for _ in range(size):
                cur = q.pop(0)
                level.append(cur)
                for i in G[cur]:
                    degree[i] -= 1
                    if degree[i] == 1: q.append(i)
        return level
```

# 1162
### As Far from Land as Possible
[Leetcode](https://leetcode.com/problems/as-far-from-land-as-possible/)


```python
class Solution(object):
    def maxDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        q = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    q.append((i, j, 0))
        dist = 0
        while q:
            i, j, dist = q.pop(0)
            for dx, dy in [(1, 0),(-1, 0),(0, -1),(0, 1)]:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 0:
                    grid[x][y] = 1
                    q.append((x, y, dist + 1))
        return dist if dist > 0 else - 1
```

# 1602
### Find Nearest Right Node in Binary Tree
[Leetcode](https://leetcode.com/problems/find-nearest-right-node-in-binary-tree/)


```python
class Solution(object):
    def findNearestRightNode(self, root, u):
        """
        :type root: TreeNode
        :type u: TreeNode
        :rtype: TreeNode
        """
        if not root: return 
        q = [root]
        while q:
            size, level = len(q), []
            for _ in range(size):
                cur = q.pop(0)
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
                level.append(cur)
            if u in level:
                ind = level.index(u)
                if ind == len(level) - 1: return 
                else: return level[ind+1]
        return
```

# 909
### Snakes and Ladders
[Leetcode](https://leetcode.com/problems/snakes-and-ladders/)


```python
class Solution(object):
    def snakesAndLadders(self, board):
        """
        :type board: List[List[int]]
        :rtype: int
        """
        def convert(square):
            i, j = (square - 1) // n, (square - 1) % n 
            if i % 2 == 0: return len(board) - i - 1, j
            else: return len(board) - i - 1, len(board[0]) - j - 1
        
        
        n = len(board)
        q = [(1, 0)]
        visited = set([1])
        while q:
            cur, step = q.pop(0)
            if cur == n * n: return step
            for i in range(1, 7):
                new = cur + i 
                if 0 < new <= n ** 2: 
                    x, y = convert(new)
                    # print(new, x, y)
                    if board[x][y] == -1:
                        if new not in visited: 
                            q.append((new, step + 1))
                            visited.add(cur)
                    elif board[x][y] not in visited:
                        # print('jump', board[x][y])
                        q.append((board[x][y], step + 1))
                        visited.add(board[x][y])
        return -1
```

# 1210
### Minimum Moves to Reach Target with Rotations
[Leetcode](https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/)


```python

```
