# 1) Write a python class to convert an integer into a roman numeral and viceversa

class Converter:
    def intToRoman(self, num):
        symbols = ["I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"]
        vals = [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        k = 12
        roman_num = ""
        while num:
            div = num // vals[k]
            num = num % vals[k]
            while div:
                div -= 1
                roman_num = roman_num + symbols[k]
            k -= 1
        
        return  roman_num

    def romanToInt(self, roman):
        symbol_dict = {"I":1,"IV":4,"V":5,"IX":9,"X":10,"XL":40,"L":50,"XC":90,"C":100,"CD":400,"D":500,"CM":900,"M":1000}
        int_val = 0
        prev_val = 1001
        for x in roman:
            cur_val = symbol_dict[x]
            if(cur_val > prev_val):
                int_val = int_val + cur_val - (prev_val * 2)
            else:
                int_val += cur_val
            prev_val = cur_val
        return int_val




obj = Converter()
n = int(input("Enter a number to convert to Roman : "))
print("Roman form of ",n," is : ",obj.intToRoman(n))

r = input("Enter a roman number to convert to int : ")
print("Integer form of ",r," is : ",obj.romanToInt(r))


