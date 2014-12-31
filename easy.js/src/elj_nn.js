(function(exports) {
    
    function LayerRegression(opt) {
        this.inp = opt.inp;
        this._grad = new Array( this.inp );

        this.forward = function(input) {
            this._act = input;
            return input;
        }.bind(this);

        this.loss = function(y) {
            var i;
            var l2sum = 0.0;
            
            this._y = y;
            for(i = 0; i < this.inp.length; i++) {
                l2sum += (y[i] - this._act[i]) * (y[i] - this._act[i]);        
            }
            return 0.5*l2sum;
        }.bind(this);

        this.grad = function() {
            for(i = 0; i < this.inp.length; i++) {
                this._grad[i] = this._act[i] - this._y[i]; 
            }
            return this._grad; 
        }.bind(this);
    };

    function NodeFullConnect(opt) {
        this.inp = opt.inp;
        if ( opt.act === "sigmoid" ) {
            this.act = function(z) {
                return 1 / (1 + Math.exp(-1*z));
            };
            this.grad = function(z) {
                var y = 1 / (1 + Math.exp(-1*z));
                return y*(1-y);
            };
        } else if ( opt.act === "tanh" ) {
            this.act = function(z) {
                var y = Math.exp(2 * x);
                return (y - 1) / (y + 1);
            };
            this.grad = function(z) {
                var y = Math.exp(2 * x);
                y =  (y - 1) / (y + 1);
                return (1 - y*y); 
            };
        }

        this.bias = 0;
        this.weights = new Array(this.inp);

        this._init = function() {
            var i;
            for(i = 0; i < this.weights.length; i++) {
                this.weights[i] = Math.random();
            } 
        }

        this.forward = function( input ) {
            var i;
            var lsum = 0.0;
            this._input = input;
            for(i = 0; i < this.weights; i++) {
                lsum += this.weights[i] * input[i];
            }    
            lsum += this.bias;
            this.z = lsum;    
            this.y = this.act(lsum);
            return this.y;
        }.bind(this);
   
        this.backward = function( grad ) {
            this.dy = grad;
            this.dz = this.dy * this.grad(this._input);
        }.bind(this); 

        this._init();
    };

    function EasyNetwork(samples) {
        
        this._init = function() {
            var i, n;
            var opt = {};
            opt.inp = 2;
            opt.act = 'sigmoid';
        
            this.hidden = [];

            var layer = [];
            opt.inp = 2;
            for(i = 0; i < 3; i++) {
                n = new NodeFullConnect(opt);
                layer.push(n); 
            }
            this.hidden.push(layer);
            layer = [];
            layer.push(new NodeFullConnect(opt));
            this.hidden.push(layer);
            
            opt.inp = 1;
            this.ouput = new LayerRegression(opt);

            
        }.bind(this);
        
        this.forward() { 
            
        }.bind(this);


        this._init();
    };
    
    exports.EasyNetwork = EasyNetwork;

})( (typeof module != 'undefined' && module.exports) || elj );
