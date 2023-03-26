class Solution {

private:
    void formBalancedParantheses(string &paranthesis, int n,vector<string> &balanced_parantheses, int open_cnt, int close_cnt) {

        if(open_cnt ==n && close_cnt ==n){
            balanced_parantheses.push_back(paranthesis);
            return;
        }
        
        if(open_cnt<n){
            paranthesis += "(";
            formBalancedParantheses(paranthesis, n, balanced_parantheses, open_cnt+1, close_cnt );
            paranthesis.pop_back();
        }

        if(close_cnt<n){
            if(open_cnt > close_cnt){
                paranthesis += ")";
                formBalancedParantheses(paranthesis, n, balanced_parantheses, open_cnt,  close_cnt+1 );
                paranthesis.pop_back();
            }
        }
    }

public:
    vector<string> generateParenthesis(int n) {
        int open_cnt=0, close_cnt=0;
        vector<string> balanced_parentheses;
        string paranthesis = "";
        formBalancedParantheses(paranthesis,n,balanced_parentheses,open_cnt,close_cnt);
        return balanced_parentheses;
    }
};