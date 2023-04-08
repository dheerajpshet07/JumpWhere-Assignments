# 19. From a list containing ints, strings and floats, make three lists to store them separately

def classify(listOfElem):
    intElem =[]
    strElem = []
    floatElem = []
    for x in listOfElem:
        if(type(x) == str):
            strElem.append(x)
        elif(type(x) == int):
            intElem.append(x)
        elif(type(x) == float):
            floatElem.append(x)
    print("Integer Elements : \n",intElem)
    print("\nFloating point Elements : \n",floatElem)
    print("\nString Elements : \n",strElem)



listOfElem = [1,2,3,4,"Hi","Hello",9,9.0,78,2,3,8.0,"uo"]
classify(listOfElem)

