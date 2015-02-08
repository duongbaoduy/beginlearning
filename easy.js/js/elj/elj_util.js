(function(exports) {
    var util = {};

    util.copySamples = function(samples) {
      var newSamples = util.zeroSamples(samples.i, samples.o, samples.d.length);

      for(var i = 0; i < samples.d.length; i++) {
        for (var j = 0; j <= samples.i; j++) {
          newSamples.d[i][j] = samples.d[i][j];
        }
      }
      return newSamples;
    };

    util.expandFeatures = function(samples, pow) {
        var i, j, k, l, v;
        var allMap = {};
        var combin = new Array(samples.i);

        for(i = 2; i <= pow; i++) {
            for(j = 0; j < samples.i; j++) {
                combin[j] = 0;
            }
            util._buildComb(allMap, combin, i);
        }
        for(i in allMap) {
            combin = JSON.parse("[" + i + "]");
            samples.i ++;
            for(j = 0; j < samples.d.length; j++) {
                v = 1.0;
                for(k = 0; k < combin.length; k++) {
                    for (l = 0; l < combin[k]; l++) {
                        v = v * samples.d[j][ samples.o + k];
                    }
                }
                samples.d[j].push(v);
            }
        }
    };

    util._buildComb = function(allMap, combin, l) {
        var i;

        if( l == 0) {
            allMap[ combin.toString() ] = 1;
            return;
        }

        for(i = 0; i < combin.length; i++) {

            combin[i] += 1;

            util._buildComb(allMap, combin, l-1);

            combin[i] -= 1;
        }

    };

    util.zeroSamples = function(i, o, number) {
      var samples = {};
      samples.o = 1;    // 0: unsupervized; 1: regssive: 2: binary class;
      samples.i = 2;
      samples.d = [];

      if (number !== undefined) {
        for(var i = 0; i < number; i++) {
          var d = [];
          if (samples.o > 0) {
            d.push(0);
          }

          for (var j = 0; j < samples.i; j++) {
            d.push(0);
          }
          samples.d.push(d);
        }
      }

      return samples;
    };

    exports.util = util;
})( (typeof module != 'undefined' && module.exports) || elj );
