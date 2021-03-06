from functools import reduce
from idlelib.tree import TreeNode
from typing import List


class Solution(object):

    # 01
    def twoSum1(self, nums, target):
        hm = {}
        for i, num in enumerate(nums):
            n = target - num
            if n not in hm:
                hm[num] = i
            else:
                return [hm[n], i]

    # 167
    def twoSum2(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i = 0
        j = len(numbers) - 1
        while i != j:
            if numbers[i] + numbers[j] == target:
                return [i, j]
            elif numbers[i] + numbers[j] < target:
                i += 1
            elif numbers[i] + numbers[j] > target:
                j -= 1

    # 653
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        nums = set()
        return self._findTarget(root, nums, k)

    def _findTarget(self, root, nums, k):
        if not root:
            return False

        if k - root.val in nums:
            return True

        nums.add(root.val)

        return self._findTarget(root.left, nums, k) or self._findTarget(root.right, nums, k)

    # 2

    class Solution(object):
        def addTwoNumbers(self, l1, l2):
            """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
            result = cur = ListNode(0)
            carry = 0

            while l1 or l2 or carry == 1:
                tempSum = 0
                if l1:
                    tempSum += l1.val
                    l1 = l1.next
                if l2:
                    tempSum += l2.val
                    l2 = l2.next
                if carry == 1:
                    tempSum += carry
                cur.next = ListNode(tempSum % 10)
                cur = cur.next
                carry = tempSum // 10
            return result.next

    # 371 calculate sum using bit
    class Solution(object):
        def getSum(self, a, b):
            """
            :type a: int
            :type b: int
            :rtype: int
            """
            mask = 0xFFFFFFFF
            if b == 0:
                return a
            tempSum = (a ^ b) & mask
            carry = ((a & b) << 1) & mask
            if (tempSum >> 31) & 1:
                return self.getSum(~(tempSum ^ mask), carry)
            else:
                return self.getSum(tempSum, carry)
        # it will get error without "& mask" for some integer like 13

    # 191
    class Solution(object):
        def hammingWeight(self, n):
            """
            :type n: int
            :rtype: int
            """
            count = 0
            while n:
                count += 1
                n &= n - 1
            return count

    # n & n - 1 means that the last bit of "1" will be removed

    # 461
    class Solution(object):
        def hammingDistance(self, x, y):
            """
            :type x: int
            :type y: int
            :rtype: int
            """
            diff = x ^ y
            count = 0
            while diff:
                diff &= diff - 1
                count += 1
            return count

    # quite same as #191, just add an & operation
    # one line method found online
    class Solution(object):
        def hammingDistance1(self, x, y):
            """
            :type x: int
            :type y: int
            :rtype: int
            """
            return bin(x ^ y).count('1')

    # bin(): converts an integer number to a binary string prefixed with 0b

    # 190 Reverse Bits
    class Solution:
        # @param n, an integer
        # @return an integer
        def reverseBits(self, n):
            temp = n
            count = 0
            result = 0
            while temp:
                if temp > (temp >> 1) << 1:
                    result += 2 ** (31 - count)
                count += 1
                temp >>= 1
            return result

    # 136
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            result = 0
            for num in nums:
                result ^= num
            return result

    # one line version1:
    class Solution(object):
        def singleNumber(self, nums):
            return reduce(lambda x, y: x ^ y, nums)

    # 137
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return (3 * sum(set(nums)) - sum(nums)) / 2

    # clever solution with bit manipulation:
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            ones = 0
            twos = 0
            for i in len(nums):
                ones = (ones ^ nums[i]) & ~ twos
                twos = (twos ^ nums[i]) & ~ ones
            return ones

    # 260
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: List[int]
            """
            res = []
            rightDiff = 1
            a = []
            b = []
            xorRes = 0
            for num in nums:
                xorRes ^= num
            while (xorRes & rightDiff == 0):
                rightDiff <<= 1
            for num in nums:
                if num & rightDiff != 0:
                    a.append(num)
                else:
                    b.append(num)
            aRes = 0
            for num in a:
                aRes ^= num
            bRes = 0
            for num in b:
                bRes ^= num
            res.append(aRes)
            res.append(bRes)
            return res

    # 7
    class Solution(object):
        def reverse(self, x):
            """
            :type x: int
            :rtype: int
            """
            if x < 0:
                sign = -1
            else:
                sign = 1
            result = sign * int(str(x)[::-1])
            return result if -(2 ** 31) - 1 < result < 2 ** 31 else 0

    # 9
    class Solution(object):
        def isPalindrome(self, x):
            """
            :type x: int
            :rtype: bool
            """
            if x < 0:
                return False
            if x < 10:
                return True
            count = 1
            while x // (10 ** count) >= 10:
                count += 1
            while x != 0:
                right = x % 10
                left = x // (10 ** count)
                if left != right:
                    return False
                x = x % (10 ** count) / 10
                count -= 2
            return True

    # 234
    class Solution(object):
        def isPalindrome(self, head):
            """
            :type head: ListNode
            :rtype: bool
            """
            count = 0
            cur = 0
            mylist = []
            while head:
                count += 1
                mylist.append(head.val)
                head = head.next
            count -= 1
            while cur <= count - cur:
                if mylist[cur] != mylist[count - cur]:
                    return False
                cur += 1
            return True

    # 14
    class Solution(object):
        def longestCommonPrefix(self, strs):
            """
            :type strs: List[str]
            :rtype: str
            """
            check = True
            count = 1
            prefix = ""
            if len(strs) > 1:
                if strs[0] != "":
                    while check:
                        if count <= len(strs[0]):
                            prefix = strs[0][0:count]
                            for str in strs:
                                if str[0:count] != prefix:
                                    check = False
                            count += 1
                        else:
                            count += 1
                            check = False
                    return prefix[0:count - 2]
                else:
                    return ""
            elif len(strs) == 1:
                return strs[0]
            else:
                return ""

    # an elegant solution found online.
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if strs:
            shortest = min(strs, key=len)
            for i in range(len(shortest)):
                if any(word[i] != shortest[i] for word in strs):
                    return shortest[:i]
            return shortest
        return ""

    # 20
    class Solution(object):
        def isValid(self, s):
            """
            :type s: str
            :rtype: bool
            """
            cheatSheet = {"{": 1, "[": 2, "(": 3, "}": 4, "]": 5, ")": 6}
            listFirstHalf = []
            listSecondHalf = []
            for i in range(len(s)):
                if cheatSheet[s[i]] < 4:
                    listFirstHalf.append(cheatSheet[s[i]])
                elif len(listFirstHalf) > 0:
                    if cheatSheet[s[i]] - 3 == listFirstHalf[-1]:
                        del listFirstHalf[-1]
                    else:
                        return False
                else:
                    return False

            if len(listFirstHalf) == 0:
                return True
            else:
                return False

    # 26 del the same list while the list is iterated
    class Solution(object):
        def removeDuplicates(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            nums.sort()
            for i in range(len(nums) - 2, -1, -1):
                if nums[i] == nums[i + 1]:
                    del nums[i]
            return len(nums)

    # 21 reverse list
    class Solution(object):
        def mergeTwoLists(self, l1, l2):
            """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
            if not l1 or not l2:
                return l1 or l2
            if l1.val < l2.val:
                l1.next = self.mergeTwoLists(l1.next, l2)
                return l1
            else:
                l2.next = self.mergeTwoLists(l1, l2.next)
                return l2

    # 53 max subarray
    class Solution:
        def maxSubArray(self, nums: List[int]) -> int:
            maxRes = nums[0]
            smallRes = 0
            for num in nums:
                if num + smallRes > 0:
                    if num >= 0:
                        smallRes += num
                        if smallRes > maxRes:
                            maxRes = smallRes
                    else:
                        smallRes += num
                else:
                    if num > maxRes:
                        maxRes = num
                    smallRes = 0
            return maxRes

    # 160
    class Solution:
        def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
            pa = headA
            pb = headB

            if pa is None or pb is None:
                return None
            while pa is not pb:
                if pa is None:
                    pa = headB
                else:
                    pa = pa.next
                if pb is None:
                    pb = headA
                else:
                    pb = pb.next
            return pa

    # 599
    class Solution:
        def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
            res = []
            resCount = 2000
            indexFirst = 0
            for str in list1:
                if str in list2:
                    temp = list2.index(str)
                    if indexFirst + temp <= resCount:
                        resCount = indexFirst + temp
                        if len(res) == 0 or indexFirst + temp <= resCount:
                            res.append(str)
                        else:
                            res[0] = str
                indexFirst += 1
            return res

    # 141

    class Solution:
        def hasCycle(self, head: ListNode) -> bool:
            try:
                faster = head.next
                slower = head
                while faster is not slower:
                    faster = faster.next.next
                    slower = slower.next
                return True
            except:
                return False

    # 876
    class Solution:
        def middleNode(self, head: ListNode) -> ListNode:
            count = 0
            resCount = head
            res = head
            while resCount:
                resCount = resCount.next
                count += 1

            for i in range(int(count / 2)):
                res = res.next
            return res

    # 206
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    # iteration
    class Solution:
        def reverseList(self, head: ListNode) -> ListNode:
            ori = head
            res = None
            while ori:
                tmp = ori.next
                ori.next = res
                res = ori
                ori = tmp
            return res

    # 92
    class Solution:
        def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
            if m == n:
                return head
            dummyNode = ListNode(0)
            dummyNode.next = head
            res = dummyNode
            resPart2 = None
            for i in range(m - 1):
                res = res.next
            ori = res.next
            for i in range(n - m + 1):
                tmp = ori.next
                ori.next = resPart2
                resPart2 = ori
                ori = tmp
            res.next.next = ori
            res.next = resPart2
            return dummyNode.next

    # 203
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def removeElements(self, head: ListNode, val: int) -> ListNode:
            dummyNode = ListNode(-1)
            dummyNode.next = head
            cur = dummyNode
            while cur is not None:
                if cur.next is not None and cur.next.val == val:
                    cur.next = cur.next.next
                else:
                    cur = cur.next
            return dummyNode.next

    # 328
    class Solution:
        def oddEvenList(self, head: ListNode) -> ListNode:
            if head is None:
                return head
            odd = head
            even = head.next
            evenHead = even
            while even and even.next:
                odd.next = odd.next.next
                even.next = even.next.next
                odd = odd.next
                even = even.next

            odd.next = evenHead
            return head

    # 67
    class Solution:
        def addBinary(self, a: str, b: str) -> str:
            if len(a) == 0:
                return b
            if len(b) == 0:
                return a
            if a[-1] == "1" and b[-1] == "1":
                return self.addBinary(self.addBinary(a[0:-1], b[0:-1]), "1") + "0"
            if a[-1] == "0" and b[-1] == "0":
                return self.addBinary(a[0:-1], b[0:-1]) + "0"
            else:
                return self.addBinary(a[0:-1], b[0:-1]) + "1"

    # 70
    class Solution:
        def climbStairs(self, n: int) -> int:
            res = [-1 for i in range(n + 1)]
            if n < 3:
                return n
            else:
                res[1] = 1
                res[2] = 2
                for i in range(3, n + 1):
                    res[i] = res[i - 1] + res[i - 2]
            return res[n]

    # 746
    class Solution:
        def minCostClimbingStairs(self, cost: List[int]) -> int:
            amount = len(cost)
            for i in range(2, amount):
                if cost[i - 1] > cost[i - 2]:
                    cost[i] += cost[i - 2]
                else:
                    cost[i] += cost[i - 1]
            if cost[amount - 1] < cost[amount - 2]:
                return cost[amount - 1]
            else:
                return cost[amount - 2]

    # 83
    class Solution:
        def deleteDuplicates(self, head: ListNode) -> ListNode:
            cur = head
            if cur:
                while cur.next:
                    if cur.val == cur.next.val:
                        cur.next = cur.next.next
                    else:
                        cur = cur.next
                return head
            else:
                return head

    # 82
    class Solution:
        def deleteDuplicates(self, head: ListNode) -> ListNode:
            dummyNode = ListNode(0)
            dummyNode.next = head
            cur = dummyNode
            val = -100
            while cur.next and cur.next.next:
                if cur.next.val == val:
                    cur.next = cur.next.next
                elif cur.next.val == cur.next.next.val:
                    val = cur.next.val
                    cur.next = cur.next.next
                else:
                    cur = cur.next
            if cur.next and cur.next.val == val:
                cur.next = cur.next.next
            return dummyNode.next

    # 100
    class Solution:
        def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
            if p and q:
                return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            return p is q

    # 101
    class Solution:
        def isSymmetric(self, root: TreeNode) -> bool:
            if root:
                if root.left and root.right:
                    if root.left.val == root.right.val:
                        return self.checkSymmetric(root.left, root.right) and self.checkSymmetric(root.left, root.right)
                    else:
                        return False
                else:
                    return root.left is root.right
            else:
                return True

        def checkSymmetric(self, subTreeA: TreeNode, subTreeB: TreeNode) -> bool:
            if subTreeA and subTreeB:
                return subTreeA.val == subTreeB.val and self.checkSymmetric(subTreeA.left,
                                                                            subTreeB.right) and self.checkSymmetric(
                    subTreeB.left, subTreeA.right)
            return subTreeA is subTreeB

    # 104
    def maxDepth(self, root: TreeNode) -> int:
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0

    # 107
    class Solution:
        def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
            level = 1
            res = []
            if root:
                temp = []
                temp.append(root.val)
                res.insert(0, temp)
                self.checkDeeper(root.left, level, res)
                self.checkDeeper(root.right, level, res)
                return res
            else:
                return root

        def checkDeeper(self, p: TreeNode, level, res):
            if p:
                level += 1
                if level > len(res):
                    temp = []
                    temp.append(p.val)
                    res.insert(0, temp)
                else:
                    res[len(res) - level].append(p.val)
                self.checkDeeper(p.left, level, res)
                self.checkDeeper(p.right, level, res)

    # 102
    class Solution:
        def levelOrder(self, root: TreeNode) -> List[List[int]]:
            level = 1
            res = []
            if root:
                temp = []
                temp.append(root.val)
                res.append(temp)
                self.checkDeeper(root.left, level, res)
                self.checkDeeper(root.right, level, res)
                return res
            else:
                return root

        def checkDeeper(self, p: TreeNode, level, res):
            if p:
                level += 1
                if level > len(res):
                    temp = []
                    temp.append(p.val)
                    res.append(temp)
                else:
                    res[level - 1].append(p.val)
                self.checkDeeper(p.left, level, res)
                self.checkDeeper(p.right, level, res)

    # 108
    def sortedArrayToBST(self, num):
        if not num:
            return None

        mid = len(num) // 2

        root = TreeNode(num[mid])
        root.left = self.sortedArrayToBST(num[:mid])
        root.right = self.sortedArrayToBST(num[mid + 1:])

        return root

    # 110
    class Solution:
        def isBalanced(self, root: TreeNode) -> bool:

            def getHeight(subRoot) -> int:
                if subRoot is None:
                    return 0
                left = getHeight(subRoot.left)
                right = getHeight(subRoot.right)

                if left == -1 or right == -1 or abs(left - right) >= 2:
                    return -1
                else:
                    return max(left, right) + 1

            return getHeight(root) != -1

    # 530
    def getMinimumDifference(self, root):
        L = []

        def dfs(node):
            if node.left: dfs(node.left)
            L.append(node.val)
            if node.right: dfs(node.right)

        dfs(root)
        return min(b - a for a, b in zip(L, L[1:]))

    # 111
    def minDepth(self, root: TreeNode) -> int:
        def nextChild(subRoot, curDepth, minDepth) -> int:
            if subRoot:
                if subRoot.left:
                    curDepth += 1
                    minDepth = nextChild(subRoot.left, curDepth, minDepth)
                    curDepth -= 1
                if subRoot.right:
                    curDepth += 1
                    minDepth = nextChild(subRoot.right, curDepth, minDepth)
                if subRoot.left is None and subRoot.right is None:
                    if minDepth == 0 or minDepth > curDepth:
                        return curDepth
            return minDepth

        return nextChild(root, 1, 0)

    # 112
    def hasPathSum(self, root, sum):
        if not root:
            return False

        if not root.left and not root.right and root.val == sum:
            return True

        sum -= root.val

        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    # 118
    class Solution:
        def generate(self, numRows: int) -> List[List[int]]:
            if numRows == 1:
                return [[1]]
            if numRows == 2:
                return [[1], [1, 1]]
            if numRows > 2:
                res = [[1], [1, 1]]
                for i in range(2, numRows):
                    tempList = [1]
                    for j in range(1, i):
                        tempList.append(res[i - 1][j - 1] + res[i - 1][j])
                    tempList.append(1)
                    res.append(tempList)
                return res

    # 119
    class Solution:
        def getRow(self, rowIndex: int) -> List[int]:
            if rowIndex == 0:
                return [1]
            if rowIndex == 1:
                return [1, 1]
            if rowIndex > 1:
                res = [[1], [1, 1]]
                for i in range(2, rowIndex + 1):
                    tempList = [1]
                    for j in range(1, i):
                        tempList.append(res[i - 1][j - 1] + res[i - 1][j])
                    tempList.append(1)
                    res.append(tempList)
                return res[rowIndex]

    # 121
    def maxProfit(self, prices: List[int]) -> int:
        diffPrices = []
        for i in range(len(prices) - 1):
            diffPrices.append(prices[i + 1] - prices[i])
        maxRes = 0
        smallRes = 0
        for num in diffPrices:
            if num + smallRes > 0:
                if num >= 0:
                    smallRes += num
                    if smallRes > maxRes:
                        maxRes = smallRes
                else:
                    smallRes += num
            else:
                if num > maxRes:
                    maxRes = num
                smallRes = 0
        return maxRes

    # 122
    def maxProfit(self, prices: List[int]) -> int:
        diffPrices = []
        for i in range(len(prices) - 1):
            diffPrices.append(prices[i + 1] - prices[i])
        maxRes = 0
        for num in diffPrices:
            if num > 0:
                maxRes += num
        return maxRes

# 123 Todo:
    def maxProfit(self, prices: List[int]) -> int:
        diffPrices = []
        for i in range(len(prices) - 1):
            diffPrices.append(prices[i + 1] - prices[i])
        firstRes = 0
        secondRes = 0
        smallRes = 0
        temp = 0
        for num in diffPrices:
            if smallRes == 0 and num > 0:
                if num > secondRes:
                    temp = num
            if num + smallRes > 0:
                if num >= 0:
                    smallRes += num
                    if smallRes > firstRes:
                        secondRes = firstRes
                        firstRes = num
                    elif smallRes > secondRes:
                        secondRes = num
                else:
                    smallRes += num
            else:
                smallRes = 0
        return firstRes + secondRes

#198:
    class Solution:
        def rob(self, nums: List[int]) -> int:
            if len(nums) == 0:
                return 0
            if len(nums) == 1:
                return nums[0]
            if (len(nums) < 3):
                return max(nums[0], nums[1])
            if (len(nums) == 3):
                return max(nums[0] + nums[2], nums[1])
            dp = [0] * len(nums)
            dp[0] = nums[0]
            dp[1] = nums[1]
            dp[2] = nums[0] + nums[2]
            for i in range(3, len(nums)):
                dp[i] = max(dp[i - 3] + nums[i], dp[i - 2] + nums[i])
            return max(dp[len(nums) - 1], dp[len(nums) - 2])

#213
    class Solution:
        def rob(self, nums: List[int]) -> int:
            if len(nums) == 0:
                return 0
            if len(nums) == 1:
                return nums[0]
            if (len(nums) < 3):
                return max(nums[0], nums[1])
            if (len(nums) == 3):
                return max(nums[0], nums[1], nums[2])
            temp = nums[len(nums)-1]
            dp0 = [0] * len(nums)
            dp0[0] = nums[0]
            dp0[1] = nums[1]
            dp0[2] = nums[0] + nums[2]
            nums[len(nums)-1] = 0
            for i in range(3, len(nums)):
                dp0[i] = max(dp0[i - 3] + nums[i], dp0[i - 2] + nums[i])
            dp1 = [0] * len(nums)
            dp1[0] = 0
            dp1[1] = nums[1]
            dp1[2] = nums[2]
            nums[len(nums) - 1] = temp
            for i in range(3, len(nums)):
                dp1[i] = max(dp1[i - 3] + nums[i], dp1[i - 2] + nums[i])
            return max(dp1[len(nums) - 1], dp1[len(nums) - 2], dp0[len(nums) - 1], dp0[len(nums) - 2])

#168
    class Solution:
        def convertToTitle(self, n: int) -> str:
            capitals = [chr(x) for x in range(ord('A'), ord('Z') + 1)]
            result = ""
            while (n > 0):
                print(n)
                result = capitals[(n - 1) % 26] + result
                n = (n - 1) // 26
            return result

#171
    class Solution:
        def titleToNumber(self, s: str) -> int:
            capitals = [chr(x) for x in range(ord('A'), ord('Z') + 1)]
            power = 0
            res = 0
            while (len(s) > 0):
                if power == 0:
                    res = capitals.index(s[-1]) + res + 1
                else:
                    res = 26 ** power * (capitals.index(s[-1]) + 1) + res
                s = s[:-1]
                power += 1
            return res

#914
    class Solution:
        def hasGroupsSizeX(self, deck: List[int]) -> bool:
            if len(deck) < 2:
                return False
            temp = {}
            smallest = 10 ^ 4
            for num in deck:
                if num in temp:
                    temp[num] += 1
                else:
                    temp[num] = 1
            for key, value in temp.items():
                if smallest > value:
                    smallest = value
            for i in range(2, smallest + 1):
                print(i)
                for key, value in temp.items():
                    if value % i != 0:
                        break
                else:
                    return True
            return False

#152
    class Solution:
        def maxProduct1(self, nums: List[int]) -> int:
            maxBefore = nums[0]
            current = 0
            maxAfter = 0
            for num in nums:
                if num == 0:
                    maxBefore = max(maxBefore, current, maxAfter)
                    print(maxBefore)
                    if maxBefore > 0:
                        current = 1
                        maxAfter = 1
                    else:
                        current = 0
                        maxAfter = 0
                else:
                    if current < 0:
                        if maxAfter == 0:
                            maxAfter = 1
                        if num > 0:
                            maxAfter *= num
                        if num < 0:
                            maxAfter = 0
                        current *= num
                    elif current > 0:
                        if num < 0:
                            print(current)
                            maxBefore = max(current, maxBefore)
                        current *= num
                    else:
                        current = num
                        maxAfter = 0

            return max(maxBefore, current, maxAfter, 0)

        class Solution:
            def maxProduct(self, nums: List[int]) -> int:
                res = nums[0]
                maxi = res
                mini = res
                for num in range(1, len(nums)):
                    if nums[num] < 0:
                        maxi, mini = mini, maxi
                    maxi = max(nums[num], nums[num] * maxi)
                    mini = min(nums[num], nums[num] * mini)

                    res = max(maxi, res)
                return res

#5
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ''
        for i in range(len(s)):
            res = max(self.check(i, i + 1, s), self.check(i, i, s), res, key=len)
        return res

    def check(self, i, j, s):
        while i >= 0 and j < len(s) and s[i] == s[j]:
            i -= 1
            j += 1
        return s[i + 1:j]

#62
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        res = [[0 for i in range(n)] for j in range(m)]
        for i in range(m):
            for j in range(n):
                if i ==0 or j ==0:
                    res[i][j] = 1
                else:
                    res[i][j] = res[i][j-1] + res[i-1][j]
        return res[m-1][n-1]

#63
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        res = [[0 for i in range(n)] for j in range(m)]
        res[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    res[i][j] = 0
                else:
                    if i == 0 and j != 0 :
                        res[i][j] = res[i][j-1]
                    elif j == 0 and i != 0:
                        res[i][j] = res[i-1][j]
                    elif j != 0 and i != 0:
                        res[i][j] = res[i][j - 1] + res[i - 1][j]
        return res[m - 1][n - 1]

#64
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = [[0 for i in range(n)] for j in range(m)]
        res[0][0] =grid[0][0]
        for i in range(m):
            for j in range(n):
                if i == 0 and j != 0:
                    res[i][j] = res[i][j - 1] + grid[i][j]
                elif j == 0 and i != 0:
                    res[i][j] = res[i - 1][j] + grid[i][j]
                elif j != 0 and i !=0:
                    res[i][j] = min(res[i-1][j], res[i][j-1]) + grid[i][j]
        return res[m - 1][n - 1]

#174
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m = len(dungeon)
        n = len(dungeon[0])
        res = [[0 for i in range(n)] for j in range(m)]
        res[0][0] = dungeon[0][0]
        smallest = [[0 for i in range(n)] for j in range(m)]
        smallest[0][0] = 1 if dungeon[0][0] > 0 else dungeon[0][0]
        for i in range(m):
            for j in range(n):
                if i == 0 and j != 0:
                    res[i][j] = res[i][j - 1] + dungeon[i][j]
                    smallest[i][j] = min(res[i][j], smallest[i][j-1])
                elif j == 0 and i != 0:
                    res[i][j] = res[i - 1][j] + dungeon[i][j]
                    smallest[i][j] = min(res[i][j], smallest[i- 1][j])
                elif j != 0 and i != 0:
                    if smallest[i - 1][j] <= smallest[i][j - 1]:
                        res[i][j] =  res[i][j - 1] + dungeon[i][j]
                        smallest[i][j] = min(smallest[i][j - 1], res[i][j])
                    else:
                        res[i][j] = res[i - 1][j] + dungeon[i][j]
                        smallest[i][j] = min(smallest[i - 1][j ], res[i][j])
        return smallest[m - 1][n - 1] if smallest[m - 1][n - 1]>0  else  smallest[m - 1][n - 1] * -1 + 1


class Solution:
    def calculateMinimumHP(self, dungeon):
        m, n = len(dungeon), len(dungeon[0])
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        dp[m - 1][n], dp[m][n - 1] = 1, 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1)

        return dp[0][0]

#202
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum([int(x) **2 for x in str(n)])
        return n == 1

#35
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if (target < nums[0]):
            return 0
        if (target > nums[len(nums) - 1]):
            return len(nums)
        left = 0
        right = len(nums) - 1
        while (True):
            if (right - left) == 1:
                return left if nums[left] == target else right
            elif (nums[int((left + right) / 2)] == target):
                return int((left + right) / 2)
            elif nums[int((left + right) / 2)] > target:
                right = int((left + right) / 2)
            elif nums[int((left + right) / 2)] < target:
                left = int((left + right) / 2)

#278
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = n-1
        l = 0
        while(l<=r):
            mid = int((r+l)/2)
            if isBadVersion(mid):
                r = mid-1
            else:
                l = mid+1
        return l

#164
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        result, count = nums[0], 0
        for num in nums:
            if num == result:
                count += 1
            elif count == 0:
                result = num
            else:
                count -= 1
        return result

#229
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        first, count1, second, count2 = 0, 0, 1, 0
        if not nums:
            return []
        for num in nums:
            if first == num:
                count1 += 1
            elif second == num:
                count2 += 1
            elif count1 == 0:
                first, count1 = num, 1
            elif count2 == 0:
                second, count2 = num, 1
            else:
                count1 -= 1
                count2 -= 1
        return [n for n in (first, second)
                    if nums.count(n) > len(nums) // 3]

#172
class Solution:
    def trailingZeroes(self, n: int) -> int:
        return 0 if n == 0 else (int)(n/5 + self.trailingZeroes(n/5))

#173
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.que = []
        self.allLeft(root)

    def next(self) -> int:
        print(self.que)
        cur = self.que.pop()
        self.allLeft(cur.right)
        return cur.val

    def hasNext(self) -> bool:
        return self.que != []

    def allLeft(self, root):
        while root:
            self.que.append(root)
            root = root.left


#175
# select p.FirstName, p.LastName, a.City, a.State from Person p left join Address a on p.PersonId = a.PersonId;

#176
#select (select distinct salary
# from Employee
# order by salary DESC
# limit 1 offset 1) as SecondHighestSalary;

#177
# CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
# BEGIN
# DECLARE M INT;
# SET M=N-1;
#   RETURN (
#       select (select distinct salary from Employee order by salary DESC limit 1 offset M) as SecondHighestSalary
#   );
# END

#303
class NumArray:

    def __init__(self, nums: List[int]):
        self.numDiff = []
        self.getNumDiff(nums)

    def sumRange(self, left: int, right: int) -> int:
        if left == 0:
            return self.numDiff[right]
        return self.numDiff[right] - self.numDiff[left - 1]

    def getNumDiff(self, nums: List[int]):
        sumFromBegin = 0
        for num in nums:
            sumFromBegin += num
            self.numDiff.append(sumFromBegin)

#304
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.sumList = []
        for row in range(len(matrix)):
            rowSum = 0
            rowList = []
            for col in range(len(matrix[0])):
                rowSum += matrix[row][col]
                rowList.append(rowSum)
            self.sumList.append(rowList)
        print(self.sumList)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        result = 0
        if col1 == 0 :
            for row in range(row1,row2+1):
                result += self.sumList[row][col2]
        else:
            for row in range(row1,row2+1):
                result = result + self.sumList[row][col2] - self.sumList[row][col1-1]
        return result

#96
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1] * dp[i-j]
        return dp[n]

#91
class Solution:

    def numDecodings(self, s: str) -> int:
        if len(s) == 0 or s[0] == "0":
            return 0
        if len(s) == 1:
            return 1
        result = 1
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2, len(s) + 1):
            if 0 < int(s[i - 1:i]):
                dp[i] += dp[i - 1]
            if s[i - 2:i][0] != '0' and int(s[i - 2:i]) <= 26:
                dp[i] += dp[i - 2]
        return dp[len(s)]

#27
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for x in nums:
            if x != val:
                nums[i] = x
                i += 1
        return i

#283
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        i = 0
        for num in nums:
            if num != 0:
                nums[i] = num
                i += 1
        while(i < len(nums)):
            nums[i] = 0
            i += 1

#1550
class Solution:
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        count = 0
        for num in arr:
            if num % 2 == 0:
                count = 0
            else:
                count += 1
                if count == 3:
                    return True
        return False

#1290
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        ans = 0
        while head:
            ans = (ans << 1) | head.val
            head = head.next
        return ans

#19
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummyNode = ListNode(0)
        dummyNode.next = head
        fastNode = dummyNode
        slowNode = dummyNode
        for i in range(n + 1):
            fastNode = fastNode.next
        while fastNode:
            slowNode = slowNode.next
            fastNode = fastNode.next
        slowNode.next = slowNode.next.next
        return dummyNode.next

#1721
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        dummyNode = ListNode(0)
        dummyNode.next = head
        firstNode = dummyNode
        secondNode = dummyNode
        firstPre = dummyNode
        secondPre = dummyNode
        fast = dummyNode
        for i in range(k - 1):
            firstPre = firstPre.next
            fast = fast.next
        firstNode = firstPre.next
        fast = fast.next
        while fast.next:
            fast = fast.next
            secondPre = secondPre.next
        secondNode = secondPre.next

        firstPre.next, secondPre.next = secondNode, firstNode
        firstNode.next, secondNode.next = secondNode.next, firstNode.next

        return dummyNode.next

#24
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        dummyNode = ListNode(0)
        dummyNode.next = head
        firstPre = dummyNode
        firstNode = dummyNode.next
        secondNode = dummyNode.next.next
        while firstNode and secondNode:
            firstPre.next, firstNode.next, secondNode.next = secondNode, secondNode.next, firstNode
            if firstNode.next is None or firstNode.next.next is None:
                return dummyNode.next
            firstPre, firstNode, secondNode = firstNode, firstNode.next, firstNode.next.next