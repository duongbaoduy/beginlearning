var elj = require('./elj_all.js');

var samples = elj.loadData('./2d.json');

elj.util.expandFeatures(samples, 7);

var logistic = new elj.Logistic(samples, 0.0001);
var sgd = new elj.SGD(10000, 0.5);

var currentValue = logistic.cost();
sgd.train(logistic, function(value) {
    console.log(">>>>>:" + value);
    if ( Math.abs(value - currentValue ) < 0.00001) {
        console.log("###############");
        return true;
    }
    currentValue = value;
    return false;
});

console.log( logistic.theta );


var coverSamples = {};
coverSamples.t = 'V';
coverSamples.i = 2;
coverSamples.o = samples.o;
coverSamples.d = [];

for (var x = -1.0; x < 1.0; x += 0.01 ) {
    for (var y = -1.0; y < 1.0; y += 0.01 ) {
        var d = [];
        d.push(-1);
        d.push(x);
        d.push(y);
        coverSamples.d.push(d);
    }
}
elj.util.expandFeatures( coverSamples, 7);

var boundSamples = {};
boundSamples.t = 'V';
boundSamples.i = 2;
boundSamples.o = 1;
boundSamples.d = [];

for ( i = 0; i < coverSamples.d.length; i++) {
    var ret = logistic.pred( coverSamples.d[i] );
    if ( Math.abs(ret-0.5) < 0.01 ) {
        var d = [];
        d.push(ret);
        d.push(coverSamples.d[i][1]);
        d.push(coverSamples.d[i][2]);
        boundSamples.d.push(d);
    }
}

elj.saveData('./bound.json', boundSamples);
