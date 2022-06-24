
def func1():
    return "func1"
    
def func2():
    return "func2"
    
def func3():
    return "func3"

def main():
    list1 = [func1, func2, func3]    
    for func in list1:
        print(func())

if __name__ == "__main__":
    main()