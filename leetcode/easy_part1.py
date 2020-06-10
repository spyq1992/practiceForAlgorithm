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
            return coun
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












solution = Solution()
print(solution.twoSum1( [1,3,6,9], 10))
print(solution.twoSum2( [1,3,6,9], 10))