class Solution {
public:
    bool isAnagram(string s, string t) {
        int sizeS = s.size();
        int sizeT = t.size();
        if((sizeS > sizeT) || (sizeS < sizeT)){
            return false;
        }
        map<char, int> freq1;
        map<char, int> freq2;

        for(int ch : s){
           freq1[ch]++;
        }

        for(int ch : t){
           freq2[ch]++;
        }

        for(int i=0;i<sizeS;i++){
            if(freq1[s[i]] != freq2[s[i]]){
                return false;
            }
        }

        return true;
    }
};