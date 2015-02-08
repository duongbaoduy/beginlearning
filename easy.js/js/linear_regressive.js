var LinearRegressive = function(d) {

  this.d = d;

  this.reset = function() {
    this.samples = [];
    this.thetas = [];

    var i;
    for(i = 0; i < this.d + 1; i++) {
        this.thetas.push(0.0);
    }
  };

  this.value = function(x) {
    var i,v;

    v = 0;
    for (i = 0; i <= this.d; i++) {
        v += this.thetas[i] * Math.pow(x, i);
    }
    return v;
  };

  this.sgd = function(maxIterate, alpha) {
    var m = this.samples.length;

    for(var i = 0; i < maxIterate; i++) {

      for(var j = 0; j < m; j++) {
        var yy = this.value( this.samples[j].x);
        var diff = alpha * ( this.samples[j].y -  yy);

        for(var d = 0; d <= this.d; d++) {
          this.thetas[d] = this.thetas[d] + diff * Math.pow( this.samples[j].x,d);
        }
      }
    }

  };


  this.reset();

};
