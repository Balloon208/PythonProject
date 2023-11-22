from flask import Flask, render_template, request, send_file
from io import BytesIO, StringIO
import numpy as np
from sympy import *
import sympy as sy
from sklearn.linear_model import LinearRegression
import pandas as df
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


weight = np.array( [ 2, 3.7, 5.4, 7.1, 8.7, 10.1 ])
minute = np.array( [ 1,2,3,4,5,6 ] )

def result():
    plt.clf()
    a = symbols('b1')
    b = symbols('b0')

    fx = sum(  ( weight - ( a*minute + b))**2 )

    pprint( sy.simplify(fx) )

    derivative1 = Derivative( fx, a).doit()
    derivative2 = Derivative( fx, b).doit()

    pprint( derivative1 )
    pprint( derivative2 )

    result = solve( (derivative1, derivative2), dict=True )
    print( result )

    x = symbols('x')
    fx2 = float(result[0][a])*x + float(result[0][b])
    pprint(fx2)

    x = symbols('x')

    fx = 2*x + 2.5
    fx2 = 1000000*x

    fx_y = []
    fx2_y = []

    for i in range( len(minute ) ) :
        fx_y.append(  fx.subs( x, minute[i] ) )
        fx2_y.append(  fx2.subs( x, minute[i] ) )

    print( fx_y )
    print( fx2_y )

    r1 = (1/4)*np.sum( ( weight - np.array(fx_y) )**2 )
    r2 = (1/4)*np.sum( ( weight - np.array(fx2_y) )**2 )

    print( 'fx : %.2f, fx2 : %.2f'%(r1, r2 ) )

    
    
    data = df.DataFrame( { "minute":minute, "loss weight":weight } )
    print(data)

    X = data['minute']
    Y = data['loss weight']
    lineFit = LinearRegression()
    lineFit.fit(X.values.reshape(-1,1), Y)
    print('기울기:', lineFit.coef_)
    print('절편:', lineFit.intercept_)


    plt.plot(X, Y, '*')
    plt.plot(X,lineFit.predict(X.values.reshape(-1,1)))
    plt.show()

    ## file로 저장하는 것이 아니라 binary object에 저장해서 그대로 file을 넘겨준다고 생각하면 됨
    ## binary object에 값을 저장한다. 
    ## svg로 저장할 수도 있으나, 이 경우 html에서 다른 방식으로 저장해줘야 하기 때문에 일단은 png로 저장해줌
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    ## object를 읽었기 때문에 처음으로 돌아가줌
    img.seek(0)
    return send_file(img, mimetype='image/png')

app = Flask(__name__)
@app.route("/")
def hello():
	return render_template('index.html')
@app.route('/test',methods=['POST'])
def post():
    global minute, weight
    minute = np.append(minute, float(request.form['minute'])) # 
    weight = np.append(weight, float(request.form['weight'])) # 
    return render_template('index.html')
@app.route('/home')
def home():
    return 'Hello, World!'
@app.route("/result",methods=['GET'])
def post2():
	return result()



if __name__ == '__main__':
    app.run(debug=True)
