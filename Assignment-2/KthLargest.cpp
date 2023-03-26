class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int> pq;
        for(auto x:nums){
            pq.push(x);
        }

        int temp = k-1;
        while(temp--){
            pq.pop();
        }
        return pq.top();
    }
};