# class Solution:
#     def threeSum(self, nums):
#         if len(nums) < 3:
#             return []
#         nums = sorted(nums)
#         left = [c for c in nums if c <= 0]
#         right = [ e for e in nums if e > 0]
#         mini = nums[0]
#         maxi = nums[-1]
#         n = list()
#         temp = ('','')
#         for num,i in enumerate(left[:-1]):
#             for j in left[num+1:]:
#                 if -i-j > maxi:
#                     continue
#                 else:
#                     for e in right[::-1]:
#                         if -i-j == e:
#                             if (i,j) == temp:
#                                 break
#                             else:
#                                 temp = (i,j)
#                             n.append([i,j,e])
#                         if -i-j > e:
#                             break
#         for num,i in enumerate(right[::-1][:-1]):
#             for j in right[::-1][num+1:]:
#                 if -i-j < mini:
#                     continue
#                 else:
#                     for e in left:
#                         if -i-j == e:
#                             if (j,i) == temp:
#                                 temp = (j,i)
#                                 break
#                             else:
#                                 temp = (j,i)
#                             n.append([e,j,i])
#                         if -i-j < e:
#                             break
#         if tuple(left[-3:]) == (0, 0, 0):
#             n.append([0,0,0])
#         return n
#
#
#
#
# if __name__ == '__main__':
#     c = [-2,0,1,1,2]
#     obj = Solution()
#     print(obj.threeSum(c))

# 语音播报模块
# import pyttsx3
# import time
#
#
# # 模块初始化
# engine = pyttsx3.init()
# print('准备开始语音播报...')
# s = time.time()
# # 设置要播报的Unicode字符串
# engine.say("Hello，人生苦短，我用Python")
# print(time.time()-s)
# # 等待语音播报完毕
# engine.runAndWait()

def sort_r(string):
    lst = string.split(",")
    res = list()
    while len(lst):
        max = lst[0]
        site = 0
        for i, s in enumerate(lst):
            if max < s and len(max) < len(s) and s[:len(max)] == max:
                continue
            if max < s:
                max = s
                site = i
        res.append(max);
        lst.pop(site)
    return ",".join(res)


if __name__ == "__main__":
    string = input()
    print(sort_r(string))

