(function (exports) {

  var SGD = function(maxIterate, step) {
    this.maxIterate = maxIterate;
    this.step = step;

    this.train = function(model, cb) {
        var targetValue = model.cost();
        var i,j,k,grad;

        for(i = 0; i < this.maxIterate; i++) {
            for(j = 0; j < model.x.length; j++) {
                grad = model.sGrad(j);
                for(k = 0; k < model.theta.length; k++) {
                    model.theta[k] +=  -1 * grad[k] * this.step;
                }
            }

            targetValue = model.cost();
            if ( cb !== undefined) {
                if ( cb(targetValue) === true) {
                    return;
                }
            }
        }
    }.bind(this);
  };

  exports.SGD = SGD;

})( (typeof module != 'undefined' && module.exports) || elj );
