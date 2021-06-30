### 总结

- 做什么题的时候会用到trie tree？
    - 字符串
    - 判断有没有出现过
    - 判断有没有前缀
    - 一个一个字母遍历
    - 需要节约空间
    - 查找前缀
- 建trie tree的注意点：
    - node有26个children， 使用ord与a的ord进行比较
    - 通常使用dfs

### 题型分类

#### 基础实现
- [Implement Trie (Prefix Tree)](#208)

#### search DFS
- [Design Add and Search Words Data Structure](#211)
- [Implement Magic Dictionary](#676)
- [Word Search II](#212)
- [Design Search Autocomplete System](#642)
- [Longest Word in Dictionary](#720)
- [Map Sum Pairs](#677)

#### not DFS
- [Replace Word](#648)


### 必备模版技巧


```python
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        now = self.root
        for ch in word:
            if ch not in now:
                now[ch] = {}
            now = now[ch]
        now['end'] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        now = self.root
        for ch in word:
            if ch not in now:
                return False
            now = now[ch]
        return now.get('end', False)

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        now = self.root
        for ch in prefix:
            if ch not in now:
                return False
            now = now[ch]
            
        return True
```

# 212
### word search ii
[Leetcode](https://leetcode.com/problems/word-search-ii/)

class TreeNode():
    def __init__(self):
        self.child = collections.defaultdict(TreeNode)
        self.isWord = False

class Trie():
    def __init__(self):
        self.root = TreeNode()
    
    def insert(self, word):
        node = self.root
        for w in word:
            node = node.child[w]
        node.isWord = True
    
    def search(self, word):
        node = self.root
        for w in word:
            node = node.children.get(w)
            if not node: return False
        return root.isWord
        

class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        res = []
        trie = Trie()
        node = trie.root
        for word in words:
            trie.insert(word)
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, node, i, j, "", res)
        return res
    
    def dfs(self, board, node, i, j, path, res):
        if node.isWord:
            res.append(path)
            node.isWord = False
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        tmp = board[i][j]
        node = node.child.get(tmp)
        if not node: return 
        board[i][j] = '#'
        self.dfs(board, node, i+1, j, path + tmp, res)
        self.dfs(board, node, i, j+1, path + tmp, res)
        self.dfs(board, node, i-1, j, path + tmp, res)
        self.dfs(board, node, i, j-1, path + tmp, res)
        board[i][j] = tmp


# 208

[Leetcode](https://leetcode.com/problems/implement-trie-prefix-tree/)

### Implement Trie (Prefix Tree)


```python
class TreeNode():
    def __init__(self):
        self.children = [None] * 26
        self.isWord = False

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TreeNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        cur = self.root
        for ch in word:
            if cur.children[ord(ch) - ord('a')] == None:
                cur.children[ord(ch) - ord('a')] = TreeNode()
            cur = cur.children[ord(ch) - ord('a')]
        cur.isWord = True 

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        cur = self.root
        for ch in word:
            if cur.children[ord(ch) - ord('a')] == None: return False
            cur = cur.children[ord(ch) - ord('a')]
        return cur.isWord

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        cur = self.root
        for ch in prefix:
            if cur.children[ord(ch) - ord('a')] == None: return False
            cur = cur.children[ord(ch) - ord('a')]
        return True   
```

# 676

[Leetcode](https://leetcode.com/problems/implement-magic-dictionary/)

###  Implement Magic Dictionary


```python
class TreeNode(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLast = False
        
class MagicDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TreeNode()

    def buildDict(self, dictionary):
        """
        :type dictionary: List[str]
        :rtype: None
        """
        for word in dictionary:
            cur = self.root
            for ch in word:
                if cur.children[ord(ch) - ord('a')] == None:
                    cur.children[ord(ch) - ord('a')] = TreeNode()
                cur = cur.children[ord(ch) - ord('a')]
            cur.isLast = True

    def search(self, searchWord):
        """
        :type searchWord: str
        :rtype: bool
        """
        return self.dfs(searchWord, 0, False, self.root)
    
    def dfs(self, word, pos, isChanged, root):
        if not root:return False
        if len(word) == pos: return isChanged and root.isLast
        flag = False
        for i in range(26):
            if ord(word[pos]) - ord('a') == i:
                flag = flag or self.dfs(word, pos + 1, isChanged, root.children[i])
            else:
                if not isChanged:
                    flag = flag or self.dfs(word, pos + 1, True, root.children[i])
                
        return flag
        
```

#  211

[Leetcode](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

### Design Add and Search Words Data Structure


```python
class TreeNode(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLast = False
        
        
class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TreeNode()


    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        cur = self.root
        for ch in word:
            if cur.children[ord(ch) - ord('a')] is None:
                cur.children[ord(ch) - ord('a')]  = TreeNode()
            cur = cur.children[ord(ch) - ord('a')]
        cur.isLast = True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """

        return self.dfs(word, self.root, 0)

    def dfs(self, word, root, pos):
        if not root: return False
        if len(word) == pos: return root.isLast
        flag = False
        if word[pos] == '.':
            for i in range(26):
                flag = flag or self.dfs(word, root.children[i], pos + 1)
        else:
            ch = root.children[ord(word[pos]) - ord('a')]
            flag = flag or self.dfs(word, ch, pos + 1)
        return flag
```

# 720
### Longest Word in Dictionary
[Leetcode](https://leetcode.com/problems/longest-word-in-dictionary/)


```python
class node(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLast = False
        
class Solution(object):
    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        root = node()
        for word in words:
            cur = root
            for ch in word:
                if not cur.children[ord(ch) - ord('a')]:
                    cur.children[ord(ch) - ord('a')] = node()
                cur = cur.children[ord(ch) - ord('a')]
            cur.isLast = True
        
        
        def dfs(root, word, count):
            if not root: return
            for i in range(26):
                if root.children[i] and root.children[i].isLast:
                    dfs(root.children[i], word + chr(ord('a') + i), count + 1)
                    if count + 1 > len(res[0]): res[0] = word + chr(ord('a') + i)
        res = ['']            
        dfs(root, '',0)
        return res[0]

```

# 648
### Replace Words
[Leetcode](https://leetcode.com/problems/replace-words/)


```python
class node(object):
    def __init__(self):
        self.children = [None] * 26
        self.isLast = False

class Solution(object):
    def replaceWords(self, dictionary, sentence):
        """
        :type dictionary: List[str]
        :type sentence: str
        :rtype: str
        """
        root = node()
        for word in dictionary:
            cur = root
            for ch in word:
                if not cur.children[ord(ch) - ord('a')]:
                    cur.children[ord(ch) - ord('a')] = node()
                cur = cur.children[ord(ch) - ord('a')]
            cur.isLast = True
        sentence = sentence.split(' ')
        res = []
        for word in sentence:
            cur = root
            i = 0
            pre = ''
            while i < len(word) and cur.children[ord(word[i]) - ord('a')] and not cur.isLast:
                pre += word[i]
                cur = cur.children[ord(word[i]) - ord('a')]
                i += 1
            if cur.isLast: res.append(pre)
            else: res.append(word)
        return ' '.join(res)
        
                    
        
```

# 677
### Map Sum Pairs
[Leetcode](https://leetcode.com/problems/map-sum-pairs/)


```python
class node(object):
    def __init__(self):
        self.children = [None] * 26
        self.count = 0
        
class MapSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = node()

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: None
        """
        cur = self.root
        for i in key:
            if not cur.children[ord(i) - ord('a')]:
                cur.children[ord(i) - ord('a')] = node()
            cur = cur.children[ord(i) - ord('a')]
        cur.count = val

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        def dfs(root):
            if not root: return 0
            res = root.count
            for i in range(26):
                if root.children[i]: res += dfs(root.children[i])
            return res
        cur = self.root
        for i in prefix:
            if cur.children[ord(i)-ord('a')]: cur = cur.children[ord(i)-ord('a')]
            else: return 0
        return dfs(cur)
            
        
        


# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)
```
