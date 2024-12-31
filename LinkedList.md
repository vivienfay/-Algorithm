# Linked List总结

## 总结

- linked list 改变结构时，就需要dummy node

- reverse list需要熟练掌握

- subarray: sum(i到j) = sum(0到j) - sum(0到i)

- 对于自身结构上的改变，需要及时存储当前结点的prev和next，以防结构改变后难以track后面的结点

- 自身结构改变的时候需要考虑是不是需要断开链以防cycle

- 看到O(nlogn)的sort基本都是merge sort或者quick sort

## 题型分类

### 建新的linked list

- [Add Two Numbers](#add-two-numbers)
- [Add Two Numbers II](#add-two-numbers-ii)

### 在原有链表上进行改动：删/并/分类

- [Merge k sorted lists](#merge-k-sorted-lists)
- [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)
- [Remove Linked List Elements](#remove-linked-list-elements)
- [Remove Duplicates from Sorted List](#remove-duplicates-from-sorted-list)
- [Remove Duplicates from Sorted List II](#remove-duplicates-from-sorted-list-ii)
- [Partition List](#partition-list)
- [Rotate List](#rotate-list)
- [Plus One Linked List](#plus-one-linked-list)

### 需要reverse链表

- [Swap Nodes in Pairs](#swap-nodes-in-pairs)
- [Reverse Nodes in k-Group](#reverse-nodes-in-k-group)
- [Reverse Linked List](#reverse-linked-list)
- [Reverse Linked List II](#reverse-linked-list-ii)
- [Reorder List](#reorder-list)

### 快慢指/ 两指针
  
- [Palindrome Linked List](#palindrome-linked-list)
- [Intersection of Two Linked Lists](#intersection-of-two-linked-lists)
- [Linked List Cycle](#linked-list-cycle)
- [Linked List Cycle II](#linked-list-cycle-ii)
- [Middle of the Linked List](#middle-of-the-linked-list)
- [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)

### 数据结构OOD

- [Copy List with Random Pointer](#copy-list-with-random-pointer)(使用hashmap做一个映射)
- [Convert Sorted List to Binary Search Tree](#convert-sorted-list-to-binary-search-tree)
- [LRU Cache](#lru-cache)

## 易错点

- 在通过两个链表组合创建新的链条的时候， 需要断开连接

- 去node.next 的时候注意一下是否存在，否则会报错

- 使用slow，fast两个指针的时候，可以都用两个head初始化

如果是单数情况，slow会落在最中点，如果是双数情况，slow会落在第二个中点

- 注意start，end，合并链表类

## 必备模版/技巧

### Reverse Linked List

- iteration

```python
    def reverse(head):
        if not head: return
        cur = head
        prev = None
        while cur.next:
            next = cur.next
            cur.next = prev
            prev, cur = cur, next
        return cur
```

- recursion

```python
# recusive

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        p = head.next
        n = self.reverseList(p)
        
        head.next = None
        p.next = head
        return n
```

### [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

- prev, start 可以一开始就定义好
- 注意prev start end怎么走的
  
```python
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head: return head
        dummy = ListNode()
        dummy.next = head
        prev, cur, start = dummy, head, dummy
        count = 1
        while count <= m:
            start, prev, cur = prev, cur, cur.next
            count += 1
        
        end = prev
        
        while count <= n:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
            count += 1
        start.next = prev
        end.next = cur
        return dummy.next
```

### [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        cur = head
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev
```

### [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)

- 注意进位时候的运算
- 注意位数不同该如何处理，两个head相同处理方式，可以用一个替代另一个去简化代码
- 注意最后位数如果需要进位如何处理

```python
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        head = cur = ListNode(-1)
        reminder = 0
        while l1 or l2 or reminder != 0:
            val = reminder
            if l1: 
                val += l1.val
                l1 = l1.next
            if l2: 
                val += l2.val
                l2 = l2.next
            reminder = val // 10
            val = val % 10
            cur.next = ListNode(val)
            cur = cur.next
        return head.next
```

### [Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)

reverse + add two number

```python
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        s1 = self.reverse(l1)
        s2 = self.reverse(l2)
        dummy = ListNode(None)
        cur = dummy
        reminder = 0 
        while s1 and s2:
            val = s1.val + s2.val + reminder
            reminder = val // 10
            val = val % 10
            cur.next = ListNode(val)
            s1, s2, cur = s1.next, s2.next, cur.next
        if s2: s1 = s2
        while s1:
            val = s1.val + reminder
            reminder = val // 10
            val = val % 10
            cur.next = ListNode(val)
            s1, cur = s1.next, cur.next
        
        if reminder != 0:
            cur.next = ListNode(reminder)
            
        return self.reverse(dummy.next)

    def reverse(self, head):
        if not head or not head.next:
            return head
        next = head.next
        new_head = self.reverse(next)
        head.next = None
        next.next = head
        return new_head
```

### [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

```python
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head: return 
        G = {}
        cur = head
        while cur:
            G[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur:
            node = G[cur]
            node.next = G[cur.next] if cur.next else None
            node.random = G[cur.random] if cur.random else None
            cur = cur.next
        return G[head]
```

### [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

- 快慢指针
- 输出node时可以slow指针从头走
- 注意判断fast.next存在

```python
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head: return False
        slow, fast = head, head.next
        while fast and fast.next:
            if slow == fast: return True
            slow, fast = slow.next, fast.next.next
        return False
```

### [Linked List Cycle ii](https://leetcode.com/problems/linked-list-cycle-ii/)

- 注意head，head next如何开始

```python
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        try:
            slow = head.next
            fast = head.next.next
            while slow is not fast:
                slow = slow.next
                fast = fast.next.next
        except:
            return None

        slow = head
        while slow is not fast:
            slow = slow.next
            fast = fast.next
            
        return slow
```

### [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

- 快慢指针

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy
        pivot = dummy
        
        for i in range(n):
            pivot = pivot.next
        
        while pivot.next:
            curr = curr.next
            pivot = pivot.next   
            
        curr.next = curr.next.next
        return dummy.next
```

### [Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

- 一层一层的断开连接
- 和82做法上的差别在于：83 一个个remove 元素, 82 套内循环，重复的链条一起remove

```python
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        prev, curr = head, head.next
        while curr:
            if curr.val == prev.val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummy.next
```

### [Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

- 快慢指针，再加一个指针标记最前面的node
- 循环里面套循环

```python
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        if not head or not head.next:
            return head
        prev, start, end = dummy, head, head.next
        while end:
            if start.val != end.val:
                prev = start
                start = end
                end = end.next
            else:
                while end and start.val == end.val:
                    end = end.next
                prev.next = end
                start = end
                if not end: break
                end = end.next
        return dummy.next    
```

### [Sort List](https://leetcode.com/problems/sort-list/)

```python
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        fast, slow, prev = head, head, head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        prev.next = None
        l1 = self.sortList(head)
        l2 = self.sortList(slow)
        return self.merge(l1, l2)
        
        
    def merge(self, l1, l2):
        new_head = ListNode(None)
        cur = new_head
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
        return new_head.next
        
```

### [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

even node后面要加none来终止死循环

```python
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        isOdd = False
        dummy = ListNode(-1)
        odd_node, even_node = head, head.next
        dummy.next = odd_node
        evenhead = even_node
        cur = head.next.next
        while cur:
            isOdd = not isOdd
            print('start',cur.val, odd_node.val, even_node.val)
            if isOdd:
                odd_node.next = cur
                odd_node = odd_node.next
            else:
                even_node.next = cur
                even_node = even_node.next
            print('end',odd_node.val, even_node.val)
            cur = cur.next
        
        even_node.next = None
        odd_node.next = evenhead    
        return dummy.next
```

### [Reorder List](https://leetcode.com/problems/reorder-list/)

- 要记得断开连接
- 考虑是以重新建一条新链，转接不同的node出发

```python
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return head
        dummy1 = ListNode(0)
        dummy1.next = head
        cur1 = head
        dummy2 = ListNode(0)
        cur2 = dummy2
        length = 0
        while cur1:
            cur2.next = ListNode(cur1.val)
            cur1 = cur1.next
            cur2 = cur2.next
            length += 1

        reverse_head = self.reverse(dummy2.next)
        dummy3 = ListNode(0)
        cur3 = dummy3
        cur1 = head
        cur2 = reverse_head
        count = 0
        while count < length:   
            cur3.next = cur1
            cur3 = cur3.next
            cur1 = cur1.next
            count += 1
            if count >= length: 
                break
            cur3.next = cur2
            cur3 = cur3.next
            cur2 = cur2.next
            count += 1

        cur3.next = None
        return dummy3.next
        
    def reverse(self, head):
        prev = None
        cur = head
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev
```

```python
# 不用复制的版本
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head or not head.next: return head
        dummy = ListNode(None)
        dummy.next = head
        prev, slow, fast = dummy, head, head
        while fast and fast.next:
            prev, slow, fast = prev.next, slow.next, fast.next.next
        prev.next = None
        new_head = self.reverse(slow)
        
        res_head = cur = ListNode(None)
        while head and new_head:
            cur.next = head
            head = head.next
            cur.next.next = new_head
            new_head = new_head.next
            cur = cur.next.next
        if head: cur.next = head
        return res_head.next

    
    def reverse(self, head):
        if not head or not head.next: return head
        next = head.next
        new_head = self.reverse(next)
        next.next = head
        head.next = None
        return new_head
```

### [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        prev, start, end = dummy, head, head
        count = 0
        while start and end:
            if count < k - 1 and end:
                end = end.next
                count += 1
            else:
                next = end.next
                end.next = None
                new_head = self.reverse(start)
                prev.next = new_head
                start.next = next
                prev = start
                start= start.next
                end = start
                count = 0
        
        return dummy.next

    def reverse(self, head):
        if not head or not head.next:
            return head
        next = head.next
        new_head = self.reverse(next)
        head.next = None
        next.next = head
        return new_head
```

### [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

```python
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        dummy.next = head
        prev, cur = dummy, head
        while cur and cur.next:
            next = cur.next
            next2 = next.next
            prev.next = next
            next.next = cur
            cur.next = next2
            prev, cur = cur, next2
        return dummy.next
```

### [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

- 使用快慢指针以及stack

```python
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        dummy = ListNode()
        cur, cur2 = head, dummy
        while cur:
            cur2.next = ListNode(cur.val)
            cur, cur2 = cur.next, cur2.next
        new_head = self.reverse(dummy.next)
        while head and new_head:
            if head.val != new_head.val: return False
            head, new_head = head.next, new_head.next
        return True

    def reverse(self, head):
        if not head or not head.next: return head
        next = head.next
        new_head = self.reverse(next)
        head.next = None
        next.next = head
        return new_head
```

```python
# Time: O(n) 
# Space: O(1) 

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head: return True
        stack = []
        slow, fast = head, head
        while fast and fast.next:
            stack.append(slow.val)
            slow, fast = slow.next, fast.next.next
        if fast: slow = slow.next
        while slow:
            cur = stack.pop()
            if cur != slow.val: return False
            slow = slow.next
        return True
```

### [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)

```python
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode()
        dummy.next = head
        prev, cur = dummy, head
        while cur:
            next = cur.next
            if cur.val == val:
                prev.next = next
                cur = next
            else:
                prev, cur = prev.next, cur.next
        return dummy.next
```

### [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

- 两个指针

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        cur1, cur2 = headA, headB
        while cur1 != cur2:
            if not cur1: cur1 = headB
            else: cur1 = cur1.next
            if not cur2: cur2 = headA
            else: cur2 = cur2.next
        return cur1
```

### Middle of the Linked List

-注意单复数

```python
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head: return
        if not head.next: return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        if not fast: return slow
        else: return slow.next
```

### [Partition List](https://leetcode.com/problems/partition-list/)

- 建两个新的链表

```python
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        if not head: return
        smaller_head = cur1 = ListNode(None)
        bigger_head = cur2 = ListNode(None)
        while head:
            if head.val >= x:
                cur2.next = head
                cur2 = cur2.next
            else:
                cur1.next = head
                cur1 = cur1.next
            head = head.next
        cur2.next = None
        cur1.next = bigger_head.next
        return smaller_head.next   
```

### [Rotate List](https://leetcode.com/problems/rotate-list/)

- 注意corner case

```python
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head: return 
        length = 0
        cur = head
        while cur:
            cur = cur.next
            length += 1
        k = length - k % length
        if length == k: return head
        dummy = ListNode(None)
        dummy.next = head
        prev, cur = dummy, head
        for i in range(k):
            prev, cur = cur, cur.next
        prev.next = None
        dummy.next = cur
        while cur.next:
            cur = cur.next
        cur.next = head
        return dummy.next       
```

### [Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

```python
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        return self.dfs(head, None)
    
    def dfs(self, left, right):
        if left == right: return 
        slow = left
        fast = left
        while fast != right and fast.next != right:
            slow, fast = slow.next, fast.next.next
        root = TreeNode(slow.val)
        root.left = self.dfs(left, slow)
        root.right = self.dfs(slow.next, right)
        return root      
```

### [LRU Cache](https://leetcode.com/problems/lru-cache/)

- 使用double linked list + hashmap
- head 和 prev是没有值的

```python
class Node(object):
    def __init__(self, key, value):
        self.key = key
        self.val = value
        self.next = None
        self.prev = None
        
class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dict = {}
        self.head = Node(None, None)
        self.tail = Node(None, None)       
        self.head.next = self.tail

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dict.keys():
            node = self.dict[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.dict.keys():
            self._remove(self.dict[key])
        n = Node(key, value)
        self._add(n)
        self.dict[key] = n
        if len(self.dict) > self.capacity:   
            n = self.tail.prev
            self._remove(n)
            del self.dict[n.key]
            
        
    def _add(self, node):
        next = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = next
        next.prev = node
        
    def _remove(self, node):
        prev = node.prev
        next = node.next
        prev.next = next
        next.prev = prev

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

### [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

```python
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if not lists: return 
        dummy = ListNode(None)
        dummy.next = lists[0]
        for head in lists[1:]:
            l1, l2 = dummy, head
            while l1.next and l2:
                if l1.next.val < l2.val: l1 = l1.next
                else:
                    next = l1.next
                    new_l2 = l2.next
                    l1.next = l2
                    l2.next = next
                    l1, l2 = l1.next, new_l2
            if l2:
                l1.next = l2
        return dummy.next
```

### [Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/submissions/)

```python
class Solution(object):
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
        new_node = ListNode(insertVal)
        if not head:
            new_node.next = new_node
            return new_node
        cur = head
        while cur:
            if cur.next.val < cur.val and (insertVal <= cur.next.val or insertVal >= cur.val):
                break
            elif cur.val <= insertVal <= cur.next.val:
                break
            elif cur.next == head:
                break
            cur = cur.next
        
        new_node.next = cur.next
        cur.next = new_node
        return head
```

### [Plus One Linked List](https://leetcode.com/problems/plus-one-linked-list/)

```python
class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        stack = []
        node = head
        while node:
            stack.append(node)
            node = node.next
        while stack:
            cur = stack.pop()
            if cur.val + 1 < 10:
                cur.val += 1
                return head
            else: cur.val = 0
        
        new_head = ListNode(1)
        new_head.next = head
        return new_head
```

### [Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/)

- 水库抽样

```python

```
