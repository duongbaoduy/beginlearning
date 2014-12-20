var elj = require('./elj_all.js');
var samples = elj.loadData('./2d.json');

var svm = new elj.SVM(samples, {C:32.0} );

svm.train(5000);

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

var boundSamples = {};
boundSamples.t = 'V';
boundSamples.i = 2;
boundSamples.o = 1;
boundSamples.d = [];

for ( i = 0; i < coverSamples.d.length; i++) {
    var ret = svm.pred( coverSamples.d[i] );
    if ( Math.abs(ret) < 0.01 ) {
        var d = [];
        d.push(ret);
        d.push(coverSamples.d[i][1]);
        d.push(coverSamples.d[i][2]);
        boundSamples.d.push(d);
    }
}

elj.saveData('./bound.json', boundSamples);
