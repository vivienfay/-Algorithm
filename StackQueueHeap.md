### 总结

#### 1. 栈

1.单调栈的用法：

找右边比当前值大的，要求连续的，可以考虑使用单调栈


### 题型分类

### stack
#### 1. stack: FILO
   
   - [Valid Parentheses](#20)
   - [Decode String](#394)
   - [Validate Stack Sequences](#946)
   - [Asteroid Collision](#735)
   - [Basic Calculator II](#227)
#### 2. stack 数据结构：
   - [Min Stack](#155)
   - [Implement Queue using Stacks](#232)
   - [Implement Stack using Queue](#225)

#### 3. 单调栈

   - [](#42)
   - [Next Greater Element II](#503)
   - [Next Greater Element I](#496)   
   - [Daily Temprature](#739)
   - [Online Stock Span](#901)
   - [](#239)
   - [Largest Rectangle in Histogram](#largest-rectangle-in-histogram)
   - []()
   - [Buildings With an Ocean View](#1762)
   - []
   
#### 4. string
   - [Minimum Remove to Make Valid Parentheses](#1249)
   

### Queue
   - [Sliding Window Maximum](#239)


#### 3. Piority Queue
   - [Top K frequent Word](#692)
   - [Furthest Building You Can Reach](#1642)
   - [Least Number of Unique Integers after K Removals](#1481)
   - [Design A Leaderboard](#1244)
   - [Course Schedule](#630)
   



### 易错点

### 必备模版技巧


```python

```

---

## Queue

# 232

[Leetcode](https://leetcode.com/problems/implement-queue-using-stacks/)

### Implement Queue using Stacks



```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.input = []
        self.output = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.input.append(x)

        

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        self.peek()
        return self.output.pop()

            
        

    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.output == []:
            while self.input != []:
                self.output.append(self.input.pop())
        return self.output[-1]
        

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.input == [] and self.output == []


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

# 225

[Leetcode](https://leetcode.com/problems/implement-stack-using-queues/)

### Implement Stack using Queues




```python
class MyStack(object):
    import collections
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = collections.deque([])

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        n = len(self.stack)
        for i in range(n-1):
            self.stack.append(self.stack.popleft())
        return self.stack.popleft()
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.stack[-1]
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return len(self.stack) == 0
```

# 155

[Leetcode](https://leetcode.com/problems/min-stack/)

### Min Stack




```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if len(self.stack) == 0:
            self.stack.append((x, x))
        else:
            tail, min_val = self.stack[-1]
            if min_val < x:
                self.stack.append((x, min_val))
            else:
                self.stack.append((x, x))

    def pop(self):
        """
        :rtype: None
        """
        return self.stack.pop()[0]

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]
        
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# 20

[Leetcode](https://leetcode.com/problems/valid-parentheses/)

### Valid Parentheses


```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        dic = {'(': ')', '{': '}', '[': ']'}
        for i in s:
            if i in dic.keys(): stack.append(i)
            if i not in dic.keys():
                try: tail = stack.pop()
                except: return False
                if i != dic[tail]: return False
        if len(stack) != 0: return False
        return True
                
```

# 844

[Leetcode](https://leetcode.com/problems/backspace-string-compare/)

### Backspace String Compare


```python
class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        stack = []
        stack_b = []
        for i in S:
            if i != '#': stack.append(i)
            elif i == '#' and stack: stack.pop()
            else: continue
        for i in T:
            if i != '#': stack_b.append(i)
            elif i == '#' and stack_b: stack_b.pop()
            else: continue
        # print(stack,stack_b)
        for i, j in zip(stack, stack_b):
            if i !=j: return False
        return len(stack) == len(stack_b)
```

# 394

[Leetcode](https://leetcode.com/problems/decode-string/)
    
### Decode String


```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        string = ''
        num_string = ''
        num_stack =[]
        alpha_stack = []
        i = 0
        while i < len(s):
            if s[i].isdigit():
                num_string += s[i]
            elif s[i].isalpha():
                string += s[i]
            elif s[i] == '[':
                num_stack.append(int(num_string))
                alpha_stack.append(string)
                string = ''
                num_string = ''
            else:
                string = alpha_stack.pop() + num_stack.pop() * string
            i += 1
        return string
```

# 1047

[Leetcode](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

### Remove All Adjacent Duplicates In String


```python
class Solution(object):
    def removeDuplicates(self, S):
        """
        :type S: str
        :rtype: str
        """
        stack = []
        for i in S:
            if stack and stack[-1] == i:
                stack.pop()
            else:
                stack.append(i)
        return ''.join(stack)
        
```

# 496 
[Leetcode](https://leetcode.com/problems/next-greater-element-i/)

### Next Greater Element I


```python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        d = {}
        st = []
        ans = []
        
        for x in nums2:
            while len(st) and st[-1] < x:
                d[st.pop()] = x
            st.append(x)

        for x in nums1:
            ans.append(d.get(x, -1))
            
        return ans
```

# 503

[Leetcode](https://leetmcode.com/problems/next-greater-element-ii/)

### Next Greater Element II


```python
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack = []
        res = [-1] * len(nums)
        for i in range(len(nums)):
            if not stack or nums[stack[-1]] > nums[i]:
                stack.append(i)
            else:
                while stack and nums[stack[-1]] < nums[i]:
                    res[stack.pop()] = nums[i]
                stack.append(i)
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                res[stack.pop()] = nums[i]
        return res
```

# 556
### Next Greater Element III
[Leetcode](https://leetcode.com/problems/next-greater-element-iii/)

- 最后check 32bit的时候 用 <<32


```python
ls = [int(i) for i in str(n)]
        for i in range(len(ls)-1, -1, -1):
            if 0 <= i - 1 and ls[i-1] < ls[i]:
                break
        if i == 0: return -1
        small = ls[i]
        ind = i
        # print(small, ind, ls[i-1])
        for x in range(i, len(ls)):
            if small > ls[x] and ls[x] > ls[i-1]: 
                small = ls[x]
                ind = x
        ls[i-1], ls[ind] = ls[ind], ls[i-1]
        res = int(''.join(str(i) for i in ls[:i] + sorted(ls[i:])))
        return res if res < 1 << 31 else -1
```

# 739

[Leetcode](https://leetcode.com/problems/daily-temperatures/)
### Daily Temperatures


```python
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        stack = []
        res = [0] * len(T)
        for i in range(len(T)):
            while stack and T[i] > T[stack[-1]]:
                ind = stack.pop()
                res[ind] = i - ind
            stack.append(i)
        return res
```

# 901

[Leetcode](https://leetcode.com/problems/online-stock-span/)

### Online Stock Span


```python
class StockSpanner(object):
    def __init__(self):
        self.stack = []

    def next(self, price):
        res = 1
        while self.stack and self.stack[-1][0] <= price:
            res += self.stack.pop()[1]
        self.stack.append([price, res])
        return res  
```

### [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

- 单调递增栈

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        stack = [-1]
        i = 0
        while i < len(heights):
            while stack[-1] != -1 and heights[stack[-1]] > heights[i]:
                value = stack.pop()
                res = max(res, heights[value] * (i - 1 - stack[-1]))
            stack.append(i)
            i += 1
        
        while stack[-1] != -1:
            ind = stack.pop()
            res = max(res, heights[ind] * (len(heights) - 1 - stack[-1]))
        return res
```

# 946

[Leetcode](https://leetcode.com/problems/validate-stack-sequences/)

### Validate Stack Sequences



```python
class Solution(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        stack = []
        i = j = 0
        while i < len(pushed):
            stack.append(pushed[i])
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
            i += 1
        return True if len(stack) == 0 else False
```

# 239

[leetcode](https://leetcode.com/problems/sliding-window-maximum/)


### Sliding Window Maximum


```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import deque
        q = deque()
        result = []
        for i in range(len(nums)):
            if q and i - q[0] == k:
                q.popleft()
            while q:
                if nums[q[-1]] < nums[i]:
                    q.pop()
                else:
                    break
            q.append(i)
            if i >= k - 1:
                result.append(nums[q[0]])
        return result
   
```

# 735
### Asteroid Collision
[Leetocde](https://leetcode.com/problems/asteroid-collision/)


```python
class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        stack = []
        for i in asteroids:
            while stack and i < 0 and stack[-1] > 0:
                if stack[-1] == -i: stack.pop(); break
                elif stack[-1] < -i: stack.pop()
                elif stack[-1] > -i: break
            else:stack.append(i)
        return stack
                    
```

# 692
### Top K Frequent Words
[Leetcode](https://leetcode.com/problems/top-k-frequent-words/)


```python
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        from collections import Counter
        import heapq
        d = Counter(words)
        ls = [(-v, key) for key, v in d.items()]
        heapq.heapify(ls)
        res = []
        for _ in range(k):
            res.append(heapq.heappop(ls)[1])
        return res
            
```

# 1249
### Minimum Remove to Make Valid Parentheses
[Leetcode](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)


```python
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack, cur = [], ''
        for i in s:
            if i == '(':
                stack.append(cur)
                cur = ''
            elif i == ')':
                if stack: cur = stack.pop()+'(' + cur + ')'
            else: cur += i
        while stack:
            cur = stack.pop() + cur
        return cur
```

### String Compression


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

# 227
### Basic Calculator II
[Leetcode](https://leetcode.com/problems/basic-calculator-ii/)


```python
class Solution:
    def calculate(self, s: str) -> int:
        if not s: return "0"
        num, stack, sign = 0, [], "+"
        s += ' '
        for i in range(len(s)):
            if s[i].isdigit():
                num = 10 * num + int(s[i])
            elif (not s[i].isdigit() and s[i] != ' ') or i == len(s) - 1:
                if sign == '-':
                    stack.append(-num)
                elif sign == '+':
                    stack.append(num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                sign = s[i]
                num = 0
        return sum(stack)
```

# 1642
### Furthest Building You Can Reach
[https://leetcode.com/problems/furthest-building-you-can-reach/]


```python
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        heap = []
        for i in range(len(heights) - 1):
            d = heights[i+1] - heights[i]
            if d > 0: heapq.heappush(heap, d)
            if len(heap) > ladders: 
                bricks -= heapq.heappop(heap)
            if bricks < 0:
                return i
        return len(heights) - 1
```

# 1481
### Least Number of Unique Integers after K Removals
[Leetcode](https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/)


```python
class Solution(object):
    def findLeastNumOfUniqueInts(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        from collections import Counter
        d = Counter(arr)
        ls = [(v, key) for key, v in d.items()]
        heapq.heapify(ls)
        for _ in range(k):
            v, k = heapq.heappop(ls)
            if v - 1 != 0: heapq.heappush(ls, (v - 1, key))
        return len(ls)
            
```


```python
# 1244
### Design A Leaderboard
[Leetcode](htmtps://leetcode.com/problems/design-a-leaderboard/)
```


      File "<ipython-input-4-702af09ce26d>", line 3
        [Leetcode](htmtps://leetcode.com/problems/design-a-leaderboard/)
                         ^
    SyntaxError: invalid syntax




```python
class Leaderboard(object):

    def __init__(self):
        self.scores = {}

    def addScore(self, playerId, score):
        """
        :type playerId: int
        :type score: int
        :rtype: None
        """
        self.scores[playerId] = self.scores.get(playerId, 0) + score

    def top(self, K):
        """
        :type K: int
        :rtype: int
        """
        h = [-score for score in self.scores.values()]
        heapq.heapify(h)
        res = 0
        for _ in range(K):
            score = heapq.heappop(h)
            res += -score
        return res

    def reset(self, playerId):
        """
        :type playerId: int
        :rtype: None
        """
        self.scores[playerId] = 0


# Your Leaderboard object will be instantiated and called as such:
# obj = Leaderboard()
# obj.addScore(playerId,score)
# param_2 = obj.top(K)
# obj.reset(playerId)
```

# 1762
### Buildings With an Ocean View
[Leetcode](https://leetcode.com/problems/buildings-with-an-ocean-view/)


```python
class Solution(object):
    def findBuildings(self, heights):
        """
        :type heights: List[int]
        :rtype: List[int]
        """
        stack = []
        for i in range(len(heights)):
            if not stack: stack.append(i)
            else:
                while stack and heights[stack[-1]] <= heights[i]:
                    stack.pop()
                stack.append(i)
        return stack
```

# 630
### Course Schedule III
[Leetcode](https://leetcode.com/problems/course-schedule-iii/)


```python
class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        courses.sort(key = lambda x: (x[1], x[0]))
        cur = 0
        pq = []
        for i, j in courses:
            cur += i
            heapq.heappush(pq, -i)
            while cur > j:
                cur += heapq.heappop(pq)            
        return len(pq)
        
```
