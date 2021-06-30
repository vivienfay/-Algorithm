###  总结

- Graph存储： 
    - Adjacency Matrix  
    - Adjacency List
        - Map
        - List

### 题型分类

- [Find the town judge](#997)
- 可用bfs或者dfs

- 有向图 有权重
- Best First Search
    - [Cheapest Flights within K steps](#787)
    - [Network Delay Time](#743)
    - [Ugly Number II](#264)
    - Find K Pairs with Smallest Sums(#373)
    - Swim in Rising Numbers(#778)
    - Kth Smallest Element in a Sorted Matrix(#378)

### 易错点


```python

```

### 必备模版技巧


```python

```

# 997

[Leetcode](https://leetcode.com/problems/find-the-town-judge/)
### Find the Town Judge

- 也可以用indegree和outdegree同时做


```python
class Solution(object):
    def findJudge(self, N, trust):
        """
        :type N: int
        :type trust: List[List[int]]
        :rtype: int
        """
        graph = {i:[] for i in range(1, N + 1)}
        indegree = {i: 0 for i in range(1, N + 1)}
        for a, b in trust:
            graph[a].append(b)
            indegree[b] += 1
        ls =[]
        for a, v in indegree.items():
            if v == N - 1: ls.append(a)
        res = -1
        for i in ls:
            if len(graph[i]) == 0: res = i
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
        G = {}
        indegree = {}
        for i, j in tickets:
            if i not in G.keys():G[i] = []
            G[i].append(j)
            indegree[j] = indegree.get(j, 0) + 1
        for i in G.keys():
            G[i].sort(reverse = True)
        res = []
        stack = ['JFK']
        while stack:
            cur = stack[-1]
            if cur in G and len(G[cur]) > 0:
                stack.append(G[cur].pop())
            else:
                res.append(stack.pop())
        return res[::-1]
```


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

# 787 
### Cheapest Flights within K steps
[Leetcode](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
- 使用piority queue


```python
class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, K):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type K: int
        :rtype: int
        """
        if not flights: return 
        from collections import defaultdict
        G = defaultdict(list)
        for i, j, distance in flights:
            G[i].append((j, distance))
        h = [(0, src, 0)]
        visited = set()
        while h:
            distance, cur, step = heappop(h)
            if cur == dst: return distance
            for destination, d in G[cur]:
                if destination not in visited and step < K + 1: 
                    heappush(h, (distance + d, destination, step+1))
        return -1
```

# 264
### Ugly Number II
[Leetcode](https://leetcode.com/problems/ugly-number-ii/)
- 还有dp的做法


```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        import heapq
        h = [1]
        order = []
        visited = set([1])
        while h:
            cur = heapq.heappop(h)
            order.append(cur)
            if len(order) == n: return order[-1]
            for i in [2, 3, 5]:
                if cur * i not in visited:
                    heapq.heappush(h, cur * i)
                    visited.add(cur * i)
```

# 743
### Network Delay Time
[Leetcode](https://leetcode.com/problems/network-delay-time/)


```python
class Solution(object):
    def networkDelayTime(self, times, N, K):
        """
        :type times: List[List[int]]
        :type N: int
        :type K: int
        :rtaype: int
        """
        from collections import defaultdict
        import heapq
        if not times: return -1
        G = defaultdict(list)
        for u, v, w in times:
            G[u].append((w, v))
        h = [(0, K)]
        dist = {}
        while h:
            time, cur = heapq.heappop(h)
            if cur in dist: continue
            dist[cur] = time
            if len(dist) == N: return time
            for dst, node in G[cur]:
                if node not in dist:
                    heapq.heappush(h, (time+dst, node))
        return -1            
         
        
                
```
