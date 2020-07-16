from functools import reduce


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

#191
    class Solution(object):
        def hammingWeight(self, n):
            """
            :type n: int
            :rtype: int
            """
            count =0
            while n:
                count +=1
                n &= n-1
            return count
    # n & n - 1 means that the last bit of "1" will be removed

#461
    class Solution(object):
        def hammingDistance(self, x, y):
            """
            :type x: int
            :type y: int
            :rtype: int
            """
            diff = x^y
            count = 0
            while diff:
                diff &= diff - 1
                count += 1
            return count
    #quite same as #191, just add an & operation
    # one line method found online
    class Solution(object):
        def hammingDistance1(self, x, y):
            """
            :type x: int
            :type y: int
            :rtype: int
            """
            return bin(x^y).count('1')
    #bin(): converts an integer number to a binary string prefixed with 0b


#190 Reverse Bits
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


#136
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
    #one line version1:
    class Solution(object):
        def singleNumber(self, nums):
            return reduce(lambda x,y:x^y, nums)

#137
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return (3*sum(set(nums)) - sum(nums))/2

    #clever solution with bit manipulation:
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
#260
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

#7
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

#9
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

#234
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

#14
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


    #an elegant solution found online.
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if strs:
            shortest = min(strs, key=len)
            for i in range(len(shortest)):
                if any(word[i] != shortest[i] for word in strs):
                    return shortest[:i]
            return shortest
        return ""
#20
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

#26 del the same list while the list is iterated
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

#21 reverse list
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

#53 max subarray
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

#160
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

#599
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        res = []
        resCount = 2000
        indexFirst = 0
        for str in list1:
            if str in list2:
                temp = list2.index(str)
                if indexFirst + temp  <= resCount:
                    resCount = indexFirst + temp
                    if len(res) == 0 or indexFirst + temp <= resCount:
                        res.append(str)
                    else:
                        res[0] = str
            indexFirst += 1
        return res

#141

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

#876
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

#206
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
#92
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

#203
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

#328
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

#67
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

#70
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


#746
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

#83
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

#82
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
#100
    class Solution:
        def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
            if p and q:
                return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            return p is q

#101
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