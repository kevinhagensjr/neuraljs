module.exports = {
    sigmoid : (x)=>{ 
        return x * (1- x); // σ (1 - σ)
    },
    tanh : (x) => { 
        return 1 - (y * y) // 1 - tanh^2(x)
    },
    relu : (x) =>{
        if(x < 0) return 0;
        else if (x > 0) return 1;
    }

};