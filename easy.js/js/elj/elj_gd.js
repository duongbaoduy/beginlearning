(function (exports) {

  var SGD = function(step) {
    this.step = step;
    
    this.batchTrain = function(model, batch, cb) {
        var targetValue;
        var i,j,k,grad;
        
        if ( cb === undefined) {
            for(j = 0; j < batch.length; j++) {
                grad = model.sGrad(batch[j]);
                for(k = 0; k < model.theta.length; k++) {
                    model.theta[k] +=  -1 * grad[k] * this.step;
                }
            }
        } else {
            for(j = 0; j < batch.length; j++) {
                grad = model.sGrad(batch[j]);
                for(k = 0; k < model.theta.length; k++) {
                    model.theta[k] +=  -1 * grad[k] * this.step;
                }
                targetValue = model.cost();
                if ( cb(targetValue) === true) {
                    break;
                } 
            }
        }
    };

    this.train = function(model, cb) {
        var targetValue;
        var i,j,k,grad;
        
        if ( cb === undefined) { 
            for(j = 0; j < model.x.length; j++) {
                grad = model.sGrad(j);
                for(k = 0; k < model.theta.length; k++) {
                    model.theta[k] +=  -1 * grad[k] * this.step;
                }
            }
        } else {
            for(j = 0; j < model.x.length; j++) {
                grad = model.sGrad(j);
                for(k = 0; k < model.theta.length; k++) {
                    model.theta[k] +=  -1 * grad[k] * this.step;
                }
                targetValue = model.cost();
                if ( cb(targetValue) === true) {
                    break;
                } 
            }
        }
    }.bind(this);

  };




  exports.SGD = SGD;

})( (typeof module != 'undefined' && module.exports) || elj );
