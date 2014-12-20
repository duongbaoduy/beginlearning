var fs = require("fs");
var elj = {};
elj.loadData = function(file) {
    var jdata = JSON.parse(fs.readFileSync(file, 'utf-8'));
    return jdata;
};

elj.saveData = function(file, samples) {
    var jdata = JSON.stringify(samples);
    fs.writeFile(file, jdata, function (err) {});    
};

elj.util = require('./elj_util.js').util;
elj.Logistic = require('./elj_logistic.js').Logistic;
elj.SGD = require('./elj_gd.js').SGD;
elj.SVM = require('./elj_svm.js').SVM;

module.exports = elj;
