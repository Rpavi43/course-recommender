import csv
import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Password@43',
    database='db1'
)

cursor = conn.cursor()

with open('sample_data_cleaned.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        cursor.execute("""
            INSERT INTO courses 
            (title, description, category, sub_category, course_type, language, skills, instructor, rating, viewers, duration, url, platform)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['title'],
            row['description'],
            row['category'],
            row['sub_category'],
            row['course_type'],
            row['language'],
            row['skills'],
            row['instructor'],
            float(row['rating']),
            int(row['viewers']),
            row['duration'],
            row['url'],
            row['platform']
        ))

conn.commit()
cursor.close()
conn.close()

print("✅ Courses inserted successfully!")
