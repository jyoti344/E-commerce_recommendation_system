from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle as pkl
import dill

app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ---------------- Load Pickles ---------------- #
main_df = pkl.load(open("./pickle_files/final_df.pkl","rb"))

book_knn = pkl.load(open("./pickle_files/book_knn.pkl","rb"))
user_knn = pkl.load(open("./pickle_files/user_knn.pkl","rb"))
fp_rules = pkl.load(open("./pickle_files/fp_rules.pkl","rb"))

user_book_recomendation = dill.load(open("./pickle_files/user_book_recomend.pkl","rb"))
search_book_recomendation = dill.load(open("./pickle_files/model_book_recomend.pkl","rb"))
also_bought = dill.load(open("./pickle_files/get_also_bought.pkl","rb"))

book_map = pkl.load(open("./pickle_files/book_map.pkl","rb"))
user_map = pkl.load(open("./pickle_files/user_map.pkl","rb"))
user_sparse = pkl.load(open("./pickle_files/user_sparse.pkl","rb"))
book_sparse = pkl.load(open("./pickle_files/book_sparse.pkl","rb"))

# Ensure user_map and book_map are categorical (like in Jupyter)
user_map = pd.Series(user_map, dtype="category")
book_map = pd.Series(book_map, dtype="category")

book_id_name = dict(enumerate(book_map.cat.categories))
user_id_name = dict(enumerate(user_map.cat.categories))
book_name_id = {v: k for k, v in book_id_name.items()}
user_name_id = {v: k for k, v in user_id_name.items()}

# ---------------- Utility Functions ---------------- #

def get_book_details(book_list):
    filtered_df = main_df[main_df['Book-Title'].isin(book_list)][['Book-Title', 'Book-Rating', 'Book-Author', 'Publisher','Image-URL-M','Book-Rating']]
    filtered_df['Book-Title'] = pd.Categorical(filtered_df['Book-Title'], categories=book_list, ordered=True)
    filtered_df = filtered_df.sort_values('Book-Title').reset_index(drop=True)
    return filtered_df



@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = int(request.form['login_id'])
        session['login_id'] = login_id
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/home')
def home():
    login_id = session.get('login_id')
    if login_id is None:
        return redirect(url_for('login'))

    index = user_name_id.get(login_id)
    if index is None:
        return "User ID not found in dataset."

    query_vector = user_sparse.getrow(index)
    distances, indices = user_knn.kneighbors(query_vector, n_neighbors=12)

    related_ids = []
    for i in range(1, len(indices.flatten())):
        idx_val = int(indices.flatten()[i])
        user = user_id_name.get(idx_val)
        if user is not None:
            related_ids.append(user)

    df = user_book_recomendation(user_id=login_id,similar_user_ids=related_ids,retings_with_book=main_df,n_recommendation=12)

    items = df.to_dict(orient='records')
    return render_template('home.html', items=items)

@app.route('/order', methods=['POST'])
def order():
    original = request.form['book_title']

    recommended_books = also_bought(original, fp_rules, top_n=5)
    df = get_book_details(recommended_books)
    also_df = df.drop_duplicates(subset=['Book-Title'])
    items = also_df.to_dict(orient='records')
    return render_template('order.html', original=original, items=items)

@app.route('/confirm', methods=['POST'])
def confirm():
    original = request.form['original_title']
    selected = request.form.getlist('selected_titles')
    final_list = [original] + selected
    df = get_book_details(final_list)
    also_df = df.drop_duplicates(subset=['Book-Title'])
    items = also_df.to_dict(orient='records')
    return render_template('confirm.html', final_list=final_list, items=items)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query'] 
    book_index = book_name_id.get(query)
    query_vector = book_sparse.getrow(book_index)
    book_distances, book_indices = book_knn.kneighbors(query_vector, n_neighbors=13)
    book_ids = search_book_recomendation(book_indices,n_recommendation=13)
    
    titles=[]
    for i in book_ids:
        titles.append(book_id_name.get(i))
        
    df = get_book_details(titles)
    df = df.drop_duplicates(subset=['Book-Title'])
    items = df.to_dict(orient='records')
    return render_template('search.html', query=query, items=items)

@app.route('/logout')
def logout():
    session.pop('login_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
