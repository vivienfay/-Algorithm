# Interval 总结

## 总结

- 解题注意点
- 解题习惯与技巧

## 题型分类

- [Interval 总结](#interval-总结)
  - [总结](#总结)
  - [题型分类](#题型分类)
    - [Merge Intervals](#merge-intervals)
    - [insert interval](#insert-interval)
    - [Meeting Room](#meeting-room)
    - [Non-overlapping Intervals](#non-overlapping-intervals)
    - [Remove Interval](#remove-interval)
    - [Interval List Intersections](#interval-list-intersections)
    - [Meeting Scheduler](#meeting-scheduler)

---------

### [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

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

### [insert interval](https://leetcode.com/problems/insert-interval/)

两种做法
- 先把新的interval插入进去，再用和merge interval一样的算法 可以直接调用 O(n)/O(n)
- 找index

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

### [Meeting Room](https://leetcode.com/problems/meeting-rooms/)

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

### [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

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

### [Remove Interval](https://leetcode.com/problems/remove-interval/)

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

### [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)

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

### [Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler/)

- merge interval类型的题，多加一个interval长度判断的条件
- 注意指针如何移动

```python
class Solution:
    def minAvailableDuration(self, slots1, slots2, duration):
        slots1 = sorted(slots1, key = lambda x: (x[0], x[1]))
        slots2 = sorted(slots2, key = lambda x: (x[0], x[1]))
        i, j = 0, 0
        while i < len(slots1) and j < len(slots2):
            a, b = slots1[i]
            c, d = slots2[j]
            interval = min(b, d) - max(a, c)
            if interval >= duration: return [max(a, c), max(a, c)+duration]
            else: 
                if b < d: i += 1
                else: j += 1
            print(a,b,c,d,i,j)
        return []
```