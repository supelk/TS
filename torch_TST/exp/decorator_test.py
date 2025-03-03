# 作者: ZY
# @Time:2025/2/24 16:46
# way down we go
def decorator_function(original_function):
    def wrapper(*args, **kwargs):
        # 这里是在调用原始函数前添加的新功能
        mod_args = [arg + 1 for arg in args]
        result = original_function(*mod_args, **kwargs)
        return result
    return wrapper


# 使用装饰器
@decorator_function
def df1(arg1, arg2):
    return arg1 + arg2  # 原始函数的实现

print(df1(1,2))