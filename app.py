from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = 'pavithra123'  # secret key for sessions

app.config.from_object(config)
mysql = MySQL(app)

# ---------------- AUTH ----------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            flash("Email already registered.", "danger")
            return redirect(url_for('signup'))

        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, hashed_password))
        mysql.connection.commit()
        cur.close()
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):  # user[3] = hashed password
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash("Welcome back, " + user[1] + "!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# ---------------- MAIN PAGES ----------------

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT title FROM courses")
    titles = [row[0] for row in cur.fetchall()]
    cur.close()
    return render_template('index.html', titles=titles)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    return render_template('dashboard.html', username=session.get('username'))

@app.route('/courses')
def courses():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, title, description, platform, url, instructor, rating FROM courses")
    data = cur.fetchall()
    cur.close()
    return render_template('courses.html', courses=data)

# ---------------- COURSE RECOMMENDER ----------------

def fetch_courses():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, title, description, skills, instructor, category, platform, url FROM courses")
    data = cur.fetchall()
    cur.close()
    return data

def recommend_courses(course_title):
    courses = fetch_courses()
    df = pd.DataFrame(courses, columns=[
        'id', 'title', 'description', 'skills', 'instructor', 'category', 'platform', 'url'
    ])

    matched_titles = df[df['title'].str.lower().str.contains(course_title.lower())]
    if matched_titles.empty:
        return pd.DataFrame()

    selected_course = matched_titles.iloc[0]
    idx = selected_course.name

    df['text'] = df['description'].fillna('') + " " + df['skills'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    course_indices = [i[0] for i in sim_scores]

    return df.iloc[course_indices]

@app.route('/recommend', methods=['POST'])
def recommend():
    keyword = request.form['keyword']
    recommended_courses = recommend_courses(keyword)

    if recommended_courses.empty:
        return render_template('recommend.html', courses=[], keyword=keyword, message="No similar courses found.")

    courses_list = recommended_courses.to_dict(orient='records')
    return render_template('recommend.html', courses=courses_list, keyword=keyword)
@app.route('/feedback', methods=['POST'])
def feedback():
    if 'user_id' not in session:
        flash("You must log in to submit feedback.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    course_id = request.form['course_id']
    rating = request.form['rating']
    feedback_text = request.form['feedback']

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO course_feedback (user_id, course_id, rating, feedback) VALUES (%s, %s, %s, %s)",
                (user_id, course_id, rating, feedback_text))
    mysql.connection.commit()
    cur.close()

    flash("Thank you for your feedback!", "success")
    return redirect(url_for('courses'))

@app.route('/save_favorite', methods=['POST'])
def save_favorite():
    if 'user_id' not in session:
        flash("You must be logged in to save favorites.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    course_id = int(request.form['course_id'])

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM favorites WHERE user_id = %s AND course_id = %s", (user_id, course_id))
    existing = cur.fetchone()

    if existing:
        flash("Already in favorites!", "info")
    else:
        cur.execute("INSERT INTO favorites (user_id, course_id) VALUES (%s, %s)", (user_id, course_id))
        mysql.connection.commit()
        flash("Course saved to favorites!", "success")

    cur.close()
    return redirect(url_for('courses'))


@app.route('/course_feedback/<int:course_id>')
def course_feedback(course_id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT u.username, f.rating, f.feedback
        FROM course_feedback f
        JOIN users u ON f.user_id = u.id
        WHERE f.course_id = %s
    """, (course_id,))
    feedback_list = cur.fetchall()
    cur.close()

    # Analyze sentiment
    analyzed = []
    for user, rating, comment in feedback_list:
        sentiment = analyze_sentiment(comment)
        analyzed.append({'user': user, 'rating': rating, 'comment': comment, 'sentiment': sentiment})

    return render_template('course_feedback.html', feedback=analyzed)




def get_cf_recommendations_pandas(current_user_id, top_n=5):
    # Step 1: Load feedback data
    cur = mysql.connection.cursor()
    cur.execute("SELECT user_id, course_id, rating FROM course_feedback")
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["user_id", "course_id", "rating"])

    # Step 2: Create user-course matrix
    rating_matrix = df.pivot_table(index='user_id', columns='course_id', values='rating').fillna(0)

    # Step 3: Compute cosine similarity
    user_sim_matrix = cosine_similarity(rating_matrix)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=rating_matrix.index, columns=rating_matrix.index)

    # Step 4: Find similar users
    if current_user_id not in user_sim_df:
        return []

    similar_users = user_sim_df[current_user_id].sort_values(ascending=False)[1:]  # skip self

    # Step 5: Weighted scores
    weighted_scores = pd.Series(dtype=np.float64)
    for other_user_id, sim_score in similar_users.items():
        other_ratings = rating_matrix.loc[other_user_id]
        weighted_scores = weighted_scores.add(other_ratings * sim_score, fill_value=0)

    # Step 6: Filter unrated courses
    user_rated = rating_matrix.loc[current_user_id]
    unrated_courses = user_rated[user_rated == 0].index
    weighted_scores = weighted_scores[unrated_courses]

    # Step 7: Get top courses
    top_courses = weighted_scores.sort_values(ascending=False).head(top_n).index.tolist()

    if not top_courses:
        return []

    # Step 8: Fetch full course info
    placeholders = ','.join(['%s'] * len(top_courses))
    cur = mysql.connection.cursor()
    cur.execute(f"SELECT id, title, description, platform, url FROM courses WHERE id IN ({placeholders})", top_courses)
    course_data = cur.fetchall()
    cur.close()

    return course_data  # [(id, title, desc, platform, url), ...]


@app.route('/recommend_cf')
def recommend_cf():
    if 'user_id' not in session:
        flash("Please log in to view personalized recommendations.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    courses = get_cf_recommendations_pandas(user_id)

    return render_template('recommend_cf.html', courses=courses, username=session['username'])

# favorites template

@app.route('/favorites')
def favorites():
    if 'user_id' not in session:
        flash("Please log in to view your saved courses.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    sort_by = request.args.get('sort', 'title')  # Default sorting

    sort_column = 'c.title' if sort_by == 'title' else 'c.platform'

    cur = mysql.connection.cursor()
    query = f"""
        SELECT c.id, c.title, c.description, c.platform, c.url
        FROM favorites f
        JOIN courses c ON f.course_id = c.id
        WHERE f.user_id = %s
        ORDER BY {sort_column}
    """
    cur.execute(query, (user_id,))
    courses = cur.fetchall()
    cur.close()

    return render_template('favorites.html', courses=courses, username=session.get('username'), sort_by=sort_by)


@app.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    if 'user_id' not in session:
        flash("You must log in first.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    course_id = request.form['course_id']

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM favorites WHERE user_id = %s AND course_id = %s", (user_id, course_id))
    mysql.connection.commit()
    cur.close()

    flash("Course removed from favorites.", "info")
    return redirect(url_for('favorites'))

@app.route('/export_favorites')
def export_favorites():
    if 'user_id' not in session:
        flash("Please log in to export.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT c.title, c.description, c.platform, c.url
        FROM favorites f
        JOIN courses c ON f.course_id = c.id
        WHERE f.user_id = %s
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()

    # Convert to CSV
    import csv
    from io import StringIO
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Title', 'Description', 'Platform', 'URL'])
    cw.writerows(rows)

    output = si.getvalue()
    return output, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=favorites.csv'
    }

def hybrid_recommendations(user_id, input_keyword, top_n=5):
    # Step 1: Content-Based
    cb_results = recommend_courses(input_keyword)
    if cb_results.empty:
        return []

    cb_df = cb_results.copy()
    cb_df['cb_score'] = [1.0 - (i / len(cb_df)) for i in range(len(cb_df))]

    # Step 2: Collaborative Filtering
    cf_courses = get_cf_recommendations_pandas(user_id, top_n=20)
    if not cf_courses:
        cb_df['hybrid_score'] = cb_df['cb_score']
        return cb_df.sort_values(by='hybrid_score', ascending=False).head(top_n).to_dict(orient='records')

    cf_df = pd.DataFrame(cf_courses, columns=['id', 'title', 'description', 'platform', 'url'])
    cf_df['cf_score'] = [1.0 - (i / len(cf_df)) for i in range(len(cf_df))]

    # Step 3: Merge
    merged = pd.merge(cb_df, cf_df[['id', 'cf_score']], on='id', how='left')
    merged['cf_score'] = merged['cf_score'].fillna(0)

    # Step 4: Final Score
    merged['hybrid_score'] = 0.5 * merged['cb_score'] + 0.5 * merged['cf_score']

    return merged.sort_values(by='hybrid_score', ascending=False).head(top_n).to_dict(orient='records')

@app.route('/recommend_hybrid', methods=['POST'])
def recommend_hybrid():
    if 'user_id' not in session:
        flash("Please log in to get personalized hybrid recommendations.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    keyword = request.form['keyword']
    courses = hybrid_recommendations(user_id, keyword)

    if not courses:
        return render_template('recommend.html', courses=[], keyword=keyword, message="No hybrid recommendations found.")

    return render_template('recommend.html', courses=courses, keyword=keyword)



def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'
# ---------------- RUN APP ----------------

if __name__ == '__main__':
    print("✅ Flask is running!")
    print(app.url_map)  # helpful for debugging routes
    app.run(debug=True)

