# E-Commerce Recommendation System

  An intelligent E-commerce Recommendation System built with Flask that suggests books using Collaborative Filtering (KNN) and Association Rule Mining (FP-Growth). 
  This project demonstrates how machine learning can be used to provide personalized recommendations, improve user experience, and simulate an e-commerce environment.

## Features
  1. Personalized Recommendations:
     When a user logs in, the system recommends books based on similar users using KNN (Collaborative Filtering).

  2. Search-Based Recommendations:
     When you search for a book, similar books are recommended based on book-to-book similarity (KNN).

  3. "Also Bought" Recommendations:
     Suggests books that are frequently bought together, generated using FP-Growth from the mlxtend library.

## Tech Stack
  
  1. Frontend: HTML, CSS (Bootstrap for styling)
  2. Backend: Flask (Python Web Framework)
  3. Machine Learning: scikit-learn, mlxtend
  4. Data Handling: pandas, numpy
  5. Visualization: matplotlib, seaborn
  6. Model Persistence: pickle, dill

## Prerequisites:
 Make sure you have the following installed before running the project:
  1. [vs_code](https://code.visualstudio.com/) – For editing and running the code
  2. [python](https://www.python.org/downloads/) - Required for running Flask and ML scripts
  3. [git](https://git-scm.com/install/windows) – (Optional) for cloning the repository
  
## Dataset:
   1. The project uses the Books Dataset from Kaggle:
      [dataset](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset)
   2. Note:
     You don’t need to download it manually — the dataset is already included inside the /books_data folder.

## Required Python Libraries:
  1. Flask
  2. scikit-learn
  3. matplotlib
  4. numpy
  5. seaborn
  6. mlxtend
  7. pickle
  8. dill
  9. pandas
  ~~~
  pip install scikit-learn matplotlib numpy seaborn mlxtend Flask pickle dill pandas
  ~~~

## Project Cloneing
   Open your terminal (or CMD) and run the following commands:
   ~~~
    # 1. Navigate to your Desktop
    cd Desktop
    # 2. Clone the repository
    git clone "https://github.com/jyoti344/E-commerce_recommendation_system.git"

    # (Or download the ZIP file manually and extract it)
   ~~~
## How to Run
  1. Make sure you are inside the project directory.
  2. Run the Flask web application:
  ~~~
  python app.py
  ~~~
  3. Open your web browser and go to: http://127.0.0.1:5000
## Login Details
  When you open the login page, enter any valid user_id from the dataset.<br>
  <br>
  Important Note:<br>
  You cannot enter a random number as a user ID.<br>
  This system uses Collaborative Filtering, which can only generate recommendations for users that existed in the training data.<br>

  You can find valid user_ids inside the ratings_with_books DataFrame in Ecommorce.ipynb.
  <br> Example:
  ~~~
   user_id = 277427
  ~~~

## How It Works
 ~~~
  |         Feature	        |       Algorithm Used	       |              Description                    |
  --------------------------------------------------------------------------------------------------------
  |User-based Recommendation|	KNN (Collaborative Filtering)|	Recommends books liked by similar users    |
  --------------------------------------------------------------------------------------------------------
  |Book-based Recommendation|	    KNN (Item Similarity)    |	Suggests similar books to the one searched |
  --------------------------------------------------------------------------------------------------------
  |   Also Bought Section   |	           FP-Growth	       |    Recommends books often bought together   |  
~~~
