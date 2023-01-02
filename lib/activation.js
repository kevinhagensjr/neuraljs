module.exports = {
    sigmoid : (x) => {
        return 1 / (1 + Math.exp(-x));
    },
    tanh : (x) => {
        return Math.tanh(x);
    },
    relu : (x) => {
        return Math.max(0,x);
    }
};