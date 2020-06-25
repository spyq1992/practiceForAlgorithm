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