from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# 파일 경로
USER_RATING_FILE = os.path.join(BASE_DIR, "output_songs_with_ids2.csv")
ITEM_VECTOR_FILE = os.path.join(BASE_DIR, "item_vectors.csv")

# 추천 함수
def recommend(bpm: int, top_n: int = 10):
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'loudness', 'speechiness', 'valence', 'mode']

    # 1. 유저 평점 불러오기
    ratings_df = pd.read_csv(USER_RATING_FILE)
    ratings_df.columns = ratings_df.columns.str.lower()
    liked_track_ids = ratings_df[ratings_df['rating'] >= 4]['track_id'].tolist()

    # 2. 아이템 벡터 불러오기
    item_df = pd.read_csv(ITEM_VECTOR_FILE)
    item_df.columns = item_df.columns.str.lower()
    item_df.drop_duplicates(inplace=True)

    # 3. 사용자 벡터 생성
    liked_items = item_df[item_df['track_id'].isin(liked_track_ids)]
    if liked_items.empty:
        return []

    user_vector = liked_items[feature_cols].mean().values.reshape(1, -1)

    # 4. BPM 필터링
    bpm_range = (bpm, bpm + 4)
    filtered_df = item_df[
        (item_df['bpm'] >= bpm_range[0]) & (item_df['bpm'] <= bpm_range[1])
    ]

    if filtered_df.empty:
        return []

    # 5. 유사도 계산
    item_vectors = filtered_df[feature_cols].values
    similarities = cosine_similarity(user_vector, item_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]

    # 6. 결과 정리
    top_items = filtered_df.iloc[top_indices].copy()
    top_items['similarity'] = similarities[top_indices]

    return [
        {
            "uri": row['track_id'],
            "bpm": round(row['bpm']),
            "similarity": round(row['similarity'], 4)
        }
        for _, row in top_items.iterrows()
    ]

@app.route('/recommend', methods=['GET'])
def recommend_api():
    bpm = int(request.args.get("bpm", 120))
    result = recommend(bpm)
    return jsonify({"tracks": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
