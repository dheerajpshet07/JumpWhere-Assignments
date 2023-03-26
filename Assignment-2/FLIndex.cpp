#include <iostream>
#include <vector>
using namespace std;

int lowerBound(vector<int>& nums, int target){
    int low = 0, high = nums.size()-1;
    int mid = 0;
    while(low < high){
        mid = low + (high-low)/2;
        if(nums[mid] < target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

int upperBound(vector<int>& nums, int target){
    int low = 0, high = nums.size()-1;
    int mid = 0;
    while(low < high){
        mid = low + (high-low)/2;
        if(nums[mid] <= target)
            low = mid + 1;
        else
            high = mid;
    }

    //to handle target coming as last element
    // if(nums[nums.size()-1]!=target)
    //     return low-1;
    // else 
        return low;
}


vector<int> searchRange(vector<int> &nums, int target){
    vector<int> vec(2, -1);
    if(nums.size() == 0){
        return vec;
    }
    int firstIndex = lowerBound(nums, target);
    if (nums[firstIndex] != target){
        return vec;
    }
    vec[0] = firstIndex;
    int lastIndex = upperBound(nums, target);

    //to handle single element
    // if(lastIndex == -1){
    //     vec[1] = firstIndex;
    // }else{
    //     vec[1] = lastIndex;
    // }

    if(nums[lastIndex] != target){
        lastIndex--;
    }
    vec[1] = lastIndex;
    return vec;
}

int main(){
    int n, target;
    cin>>n;
    vector<int> vec(n);
    for(int i=0;i<n;i++){
        cin>>vec[i];
    }
    cin>>target;
    vector<int> tVec = searchRange(vec,target);
    for(int i=0;i<2;i++){
        cout<<tVec[i]<<" ";
    }
    return 0;
}