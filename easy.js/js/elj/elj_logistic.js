(function (exports) {
    "use strict";
    var Logistic = function(samples, lambda) {
      this.theta = [];
      this.x = [];
      this.y = [];

      if ( lambda !== undefined ) {
          this.lambda = lambda;
      }

      this._init = function() {
        for(var i = 0; i < samples.i + 1; i++) {
          this.theta.push(0.0);
        }
        for(var i = 0; i < samples.d.length; i++) {
          var x = [];
          x.push(1.0);
          x = x.concat( samples.d[i].slice(1));
          this.x.push(x);
          this.y.push( samples.d[i][0] );
        }
      }.bind(this);

      this.pred = function(sample) {
        var x = [];
        x.push(1.0);
        x = x.concat( sample.slice(1) );
        var l = this._sigmod( this._linear(x) );
        return l;
      }.bind(this);

      this.cost = function() {
        var sumCost = 0.0;

        for(var i = 0; i < this.x.length; i++) {
          var l = this._sigmod( this._linear(this.x[i]) );
          if ( this.y[i] === 1) {
              sumCost =  sumCost - Math.log(l);
          } else {
              sumCost = sumCost - Math.log( 1 - l);
          }
        }

        sumCost = sumCost / this.x.length;

        if ( this.lambda !== undefined) {
            for(var i = 1; i < this.theta.length; i++) {
                sumCost += this.lambda * this.theta[i] * this.theta[i] / 2 ;
            }
        }

        return sumCost;
      }.bind(this);

      this.sGrad = function(i) {
        var l = this._sigmod( this._linear(this.x[i]) );
        var gtheta = [];
        for(var j = 0; j < this.theta.length; j++) {
          gtheta.push( this.x[i][j] * (l - this.y[i]) / this.x.length );
          if ( this.lambda !== undefined && j > 0) {
            gtheta[j] += this.lambda * this.theta[j];
          }
        }
        return gtheta;
      }.bind(this);

      this._linear = function(x) {
        var i;
        var sum = 0.0;
        for(i = 0; i < this.theta.length; i++) {
          sum = sum + this.theta[i] * x[i];
        }
        return sum;
      }.bind(this);


      this._sigmod = function(z) {
        return 1.0 / (1 + Math.exp(-1*z) );
      };

      this._init();
    };
    exports.Logistic = Logistic;

})( (typeof module != 'undefined' && module.exports) || elj );
