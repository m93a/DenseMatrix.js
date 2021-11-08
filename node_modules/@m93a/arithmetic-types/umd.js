(function(define) {
    define(function (require, exports, module) {
        module.exports = require('./lib/umd');
    });
}(
    typeof module === 'object' && module.exports && typeof define !== 'function' ?
    function (factory) { module.exports = factory(require, exports, module); } :
    define
));
