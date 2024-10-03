import base64
import datetime
import hashlib
from io import BytesIO
from flask import (Flask, render_template, request, jsonify, redirect, session)
from werkzeug.security import check_password_hash
from functools import lru_cache, wraps
from connect import db, cursor as db_cursor
import math
import joblib
import pandas as pd
import os
from lbsa.sentistrength_id import SentiStrengthID
senti = SentiStrengthID()
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import joblib
from flask_paginate import Pagination, get_page_args

app = Flask(__name__)

#enkripsi session
app.secret_key = 'sbdfd3223yfh8bdniff8w'

#fungsi dekorator untuk mengecek apakah user sudah login
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('login')
    return wrap

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations')
def recommendation():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    tag = request.args.get('tag')

    page = request.args.get('page', type=int, default=1)
    products = get_recommendations(age_range, skin_type, tag)

    pagination = Pagination(page=page, total=len(products), record_name='products', per_page=10)

    products = products[(page - 1) * 10 : min(page * 10, len(products))]
    return render_template('recommendation.html', products=products, age_range=age_range, skin_type=skin_type, tag=tag, pagination=pagination)

@lru_cache(maxsize=32)
def get_recommendations(age_range, skin_type, tag):
    cursor = db_cursor(dictionary=True)
    sql = "SELECT name from training_results where tag=%s and active=1"
    val = (tag,)
    cursor.execute(sql,val)
    data = cursor.fetchone()
    if data['name'] == 'default':
        name = tag
    else:
        name = data['name'][:8]

    sql = "SELECT products.*, count(reviews.id) as review_count from products join reviews on products.id=reviews.product_id  where tag=%s and reviews.age_range=%s and reviews.skin_type like %s group by products.id"
    val = (tag, age_range, f"{skin_type}%")

    cursor.execute(sql,val)
    data = cursor.fetchall()
    data = {row['id']: row for row in data}

    data_input= []
    product_id= list(data.keys())
    for id in product_id:
        data_input.append({
            "product_id":id,
            "age_range":age_range,
            "skin_type":skin_type
        })

    data_encoding = []
    for inputan in data_input:
        data_encoding.append({
            'product_id': inputan["product_id"],
            'age_range_18 and Under': 1 if inputan["age_range"]== "18 and Under" else 0,
            'age_range_19 - 24': 1 if inputan["age_range"]== "19 - 24" else 0,
            'age_range_25 - 29': 1 if inputan["age_range"]== "25 - 29" else 0,
            'age_range_30 - 34': 1 if inputan["age_range"]== "30 - 34" else 0,
            'age_range_35 - 39': 1 if inputan["age_range"]== "35 - 39" else 0,
            'age_range_40 - 44': 1 if inputan["age_range"]== "40 - 44" else 0,
            'age_range_45 and Above': 1 if inputan["age_range"]== "45 and Above" else 0,
            'skin_type_Combination': 1 if inputan["skin_type"]== "Combination" else 0,
            'skin_type_Dry': 1 if inputan["skin_type"]== "Dry" else 0,
            'skin_type_Normal': 1 if inputan["skin_type"]== "Normal" else 0,
            'skin_type_Oily': 1 if inputan["skin_type"]== "Oily" else 0,
        })
    
    X = pd.DataFrame(data_encoding)

    dtr1 = joblib.load(f"models/dtr1/{name}.pkl")
    rating = dtr1.predict(X)

    dtr2 = joblib.load(f"models/dtr2/{name}.pkl")
    rating_text = dtr2.predict(X)

    X2 = pd.DataFrame({
    "rating":rating,
    "rating_text":rating_text,
    })

    svm = joblib.load(f"models/svm/{name}.pkl")
    is_recommended = svm.predict(X2)

    hasil=data.copy()
    for id1, rating1, rating_text1, is_recommended1 in zip(product_id, rating, rating_text, is_recommended):
        if is_recommended1 == -1:
            del hasil[id1]
            continue
        hasil[id1].update({
            'rating':rating1,
            'rating_text':rating_text1,
            'average':(rating1+rating_text1)/2,
            'is_recommended':is_recommended1
        })
    for i in hasil.keys():
        hasil[i]['final_score']=0 if hasil[i]['review_count']==0 else wilson_score(hasil[i]['average'], hasil[i]['review_count'])*5

    hasil_akhir = [hasil[i] for i in hasil.keys() if hasil[i]['review_count']!= 0]
    hasil_akhir.sort(key=lambda row: row['final_score'], reverse=True)

    return hasil_akhir

def wilson_score(average,review_count):
    p= average/5
    n= review_count
    z= 1.96
    a= p+z**2/(2*n)
    b= math.sqrt((p*(1-p) + z**2/4*n)/n)
    c= 1 + z**2 /n
    atas= (a+b)/c
    bawah= (a-b)/c

    return (atas+bawah)/2

@app.route('/reviews')
def reviews():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    product_id = request.args.get('product_id')
    cursor = db_cursor(dictionary=True)
    sql = "SELECT reviews.* from reviews where reviews.product_id=%s and reviews.age_range=%s and reviews.skin_type like %s"
    val = (product_id, age_range, f"{skin_type}%")
    cursor.execute(sql,val)
    data = cursor.fetchall()
    return jsonify(data)

@app.route('/process')
def process():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    tag = request.args.get('tag')
    cursor = db_cursor(dictionary=True)
    sql = "SELECT name from training_results where tag=%s and active=1"
    val = (tag,)
    cursor.execute(sql,val)
    data = cursor.fetchone()
    if data['name'] == 'default':
        name = tag
    else:
        name = data['name'][:8]

    sql = "SELECT products.*, count(reviews.id) as review_count from products join reviews on products.id=reviews.product_id  where tag=%s and reviews.age_range=%s and reviews.skin_type like %s group by products.id"
    val = (tag,age_range,f"{skin_type}%")

    cursor.execute(sql,val)
    data = cursor.fetchall()
    data = {row['id']: row for row in data}
    # Mengambil data produk
    df_awal = pd.DataFrame(data.values()).to_html(columns=["id","name","brand"])
    
    data_input= []
    product_id= list(data.keys())
    for id in product_id:
        data_input.append({
            "product_id":id,
            "age_range":age_range,
            "skin_type":skin_type
        })
    # Membuat data input dengan menggabungkan product_id, skin_type, dan age_range
    df_input = pd.DataFrame(data_input).to_html()

    data_encoding = []
    for inputan in data_input:
        data_encoding.append({
            'product_id': inputan["product_id"],
            'age_range_18 and Under': 1 if inputan["age_range"]== "18 and Under" else 0,
            'age_range_19 - 24': 1 if inputan["age_range"]== "19 - 24" else 0,
            'age_range_25 - 29': 1 if inputan["age_range"]== "25 - 29" else 0,
            'age_range_30 - 34': 1 if inputan["age_range"]== "30 - 34" else 0,
            'age_range_35 - 39': 1 if inputan["age_range"]== "35 - 39" else 0,
            'age_range_40 - 44': 1 if inputan["age_range"]== "40 - 44" else 0,
            'age_range_45 and Above': 1 if inputan["age_range"]== "45 and Above" else 0,
            'skin_type_Combination': 1 if inputan["skin_type"]== "Combination" else 0,
            'skin_type_Dry': 1 if inputan["skin_type"]== "Dry" else 0,
            'skin_type_Normal': 1 if inputan["skin_type"]== "Normal" else 0,
            'skin_type_Oily': 1 if inputan["skin_type"]== "Oily" else 0,
        })
    # Melakukan one hot encoding
    df_encoding = pd.DataFrame(data_encoding).to_html()

    X = pd.DataFrame(data_encoding)

    dtr1 = joblib.load(f"models/dtr1/{name}.pkl")
    rating = dtr1.predict(X)

    df_rating = X.copy().assign(rating=rating).to_html()

    dtr2 = joblib.load(f"models/dtr2/{name}.pkl")
    rating_text = dtr2.predict(X)

    df_rating_text = X.copy().assign(rating_text=rating_text).to_html()

    X2 = pd.DataFrame({
    "rating":rating,
    "rating_text":rating_text,
    })

    svm = joblib.load(f"models/svm/{name}.pkl")
    is_recommended = svm.predict(X2)
    
    df_is_recommended = X2.copy().assign(is_recommended=is_recommended).to_html()

    hasil=data.copy()
    for id1, rating1, rating_text1, is_recommended1 in zip(product_id, rating, rating_text, is_recommended):
        if is_recommended1 == -1:
            del hasil[id1]
            continue
        hasil[id1].update({
            'rating':rating1,
            'rating_text':rating_text1,
            'average':(rating1+rating_text1)/2,
            'is_recommended':is_recommended1
        })
    df_hasil = pd.DataFrame(hasil.values()).to_html(columns=["id","rating","rating_text","average"])

    for i in hasil.keys():
        hasil[i]['final_score']=0 if hasil[i]['review_count']==0 else wilson_score(hasil[i]['average'], hasil[i]['review_count'])*5
    df_final_score = pd.DataFrame(hasil.values()).to_html(columns=["id","average","review_count","final_score"])

    hasil_akhir = [hasil[i] for i in hasil.keys() if hasil[i]['review_count']!= 0]
    hasil_akhir.sort(key=lambda row: row['final_score'], reverse=True)
    df_hasil_akhir = pd.DataFrame(hasil_akhir).to_html(columns=["id","name","brand","price","final_score"])
    return render_template('process.html', df_awal=df_awal, df_input=df_input, df_encoding=df_encoding, df_rating=df_rating, df_rating_text=df_rating_text, df_is_recommended=df_is_recommended, df_hasil=df_hasil, df_final_score=df_final_score, df_hasil_akhir=df_hasil_akhir)

@app.route('/login', methods=['POST', 'GET'])
def login():
    if 'logged_in' in session:
        return redirect('admin')
    if request.method == 'POST':
        password = request.form['password']
        kursor = db_cursor(dictionary=True)
        sql = "SELECT value from settings where name='ADMIN_PASSWORD'"
        kursor.execute(sql)
        data = kursor.fetchone()
        if check_password_hash(data['value'], password):
            session['logged_in'] = True
            return redirect('admin')
        else:
            return render_template('login.html', error='Password salah')
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('login')

@app.route('/admin')
@login_required
def admin():
    return render_template('admin.html')

@app.route('/models')
@login_required
def models():
    tag = request.args.get('tag')
    kursor = db_cursor(dictionary=True)
    sql = "SELECT * from training_results where tag=%s"
    val = (tag,)
    kursor.execute(sql,val)
    data = kursor.fetchall()
    return render_template('models.html', data=data, tag=tag)

@app.route('/model/active')
@login_required
def model_active():
    tag = request.args.get('tag')
    name = request.args.get('name')
    kursor = db_cursor(dictionary=True)
    sql = "UPDATE training_results set `active`='1' where `name`=%s and tag=%s"
    val = (name,tag)
    kursor.execute(sql,val)
    sql = "UPDATE training_results set `active`='0' where `name`<>%s and tag=%s"
    val = (name,tag)
    kursor.execute(sql,val)
    db.commit()
    return redirect('/models?tag='+tag)

@app.route('/model/delete/<id>')
@login_required
def model_delete(id):
    kursor = db_cursor(dictionary=True)
    sql = "SELECT name from training_results where id=%s"
    val = (id,)
    kursor.execute(sql,val)
    data = kursor.fetchone()
    name = data['name'][:8]
    sql = "DELETE from training_results where id=%s"
    val = (id,)
    kursor.execute(sql,val)
    db.commit()
    for model in ['dtr1','dtr2','svm']:
        try:
            os.remove(f"models/{model}/{name}.pkl")
        except:
            pass
    return redirect(request.referrer)

@app.route('/model/<id>')
@login_required
def model(id):
    kursor = db_cursor(dictionary=True)
    sql = "SELECT * from training_results where id=%s"
    val = (id,)
    kursor.execute(sql,val)
    data = kursor.fetchone()
    return render_template('model.html', data=data)

@app.route('/training', methods=['POST'])
@login_required
def training():
    tag = request.form['tag']
    dataset = request.files['dataset']
    dataset.save("datasets.csv")

    checksum = hashlib.md5(open('datasets.csv', 'rb').read()).hexdigest()
    kursor = db_cursor(dictionary=True)
    sql = "SELECT * from training_results where name=%s"
    val = (checksum,)
    kursor.execute(sql,val)
    data = kursor.fetchone()
    if data:
        return redirect('/model/'+str(data['id']))
    
    raw_data = pd.read_csv("datasets.csv")

    cleaned_data = raw_data.dropna()
    cleaned_data = cleaned_data[cleaned_data.is_recommended != 0]
    cleaned_data.skin_type.replace(" Skin", "", inplace=True, regex=True)

    def process(row):
        row['rating_text'], row['processed_text'] = senti.score(row['text'])
        return row

    processed_data = cleaned_data.apply(process, axis=1)

    X = processed_data[['product_id', 'age_range', 'skin_type']]
    X = pd.get_dummies(X)
    y = processed_data['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/dtr1/{checksum[:8]}.pkl")

    dtr1_train = X_train.merge(y_train, left_index=True, right_index=True)

    dtr1_test = X_test.merge(y_test, left_index=True, right_index=True)

    y_pred = model.predict(X_test)
    dtr1_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    r2_1 = r2_score(y_test, y_pred)
    mae1 = mean_absolute_error(y_test, y_pred)

    X = processed_data[['product_id', 'age_range', 'skin_type']]
    y = processed_data['rating_text']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/dtr2/{checksum[:8]}.pkl")

    dtr2_train = X_train.merge(y_train, left_index=True, right_index=True)

    dtr2_test = X_test.merge(y_test, left_index=True, right_index=True)

    y_pred = model.predict(X_test)
    dtr2_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    r2_2 = r2_score(y_test, y_pred)
    mae2 = mean_absolute_error(y_test, y_pred)

    X = processed_data[['rating', 'rating_text']]
    y = processed_data['is_recommended']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/svm/{checksum[:8]}.pkl")

    svm_train = X_train.merge(y_train, left_index=True, right_index=True)

    svm_test = X_test.merge(y_test, left_index=True, right_index=True)

    y_pred = model.predict(X_test)
    svm_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp = disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix')
    gambar_cm = BytesIO()
    plt.savefig(gambar_cm, format='png')

    sql = """
    INSERT into training_results (`tag`, `name`, `r2_1`, `mae1`, `r2_2`, `mae2`, `accuracy`, `precision`, `recall`, `f1_score`,
    `confusion_matrix`,
    `raw_data`, `cleaned_data`, `processed_data`,
    `dtr1_train`, `dtr1_test`, `dtr1_compare`,
    `dtr2_train`, `dtr2_test`, `dtr2_compare`,
    `svm_train`, `svm_test`, `svm_compare`,
    `train_date`, `active`)
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
    """
    val = (tag, checksum, r2_1, mae1, r2_2, mae2, accuracy, precision, recall, f1_score,
            base64.encodebytes(gambar_cm.getvalue()).decode('utf-8'),
            raw_data.head().to_html(header="true").replace('border="1"', ''),
            cleaned_data.head().to_html(header="true").replace('border="1"', ''),
            processed_data.head().to_html(header="true").replace('border="1"', ''),
            dtr1_train.head().to_html(header="true").replace('border="1"', ''),
            dtr1_test.head().to_html(header="true").replace('border="1"', ''),
            dtr1_compare.head().to_html(header="true").replace('border="1"', ''),
            dtr2_train.head().to_html(header="true").replace('border="1"', ''),
            dtr2_test.head().to_html(header="true").replace('border="1"', ''),
            dtr2_compare.head().to_html(header="true").replace('border="1"', ''),
            svm_train.head().to_html(header="true").replace('border="1"', ''),
            svm_test.head().to_html(header="true").replace('border="1"', ''),
            svm_compare.head().to_html(header="true").replace('border="1"', ''),
            datetime.datetime.now())
    kursor.execute(sql,val)
    db.commit()
    id = kursor.lastrowid
    return redirect('/model/'+str(id))

